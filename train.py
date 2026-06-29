import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from config import OUTPUT_DIR
from data_loader import load_data, build_datasets, build_val_or_test_loader
from model import build_multi_input_model
from metrics import evaluate_and_save_metrics

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Input Bone Age Model with PyTorch")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick test with 1% of data and 2 epochs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--seed", type=int, default=0, help="Torch init/training seed (data split stays fixed)")
    parser.add_argument("--backbone", type=str, default="efficientnet_b3", help="Backbone to use")
    args = parser.parse_args()

    # Seed torch only -> isolates model-init/training stochasticity.
    # Data split is fixed by RANDOM_STATE in data_loader, so all seeds see the same data.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sample_frac = 0.01 if args.quick_test else 1.0
    epochs = 2 if args.quick_test else args.epochs
    backbone_name = args.backbone
    run_name = f"seed{args.seed}" + ("_quicktest" if args.quick_test else "")

    # All outputs for this run live under OUTPUT_DIR/<run_name>/
    run_dir = OUTPUT_DIR / backbone_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Backbone: {backbone_name}] Starting run: {run_name}")
    print(f"Using {sample_frac*100}% of data for {epochs} epochs.")

    print("Loading data...")
    train_df, val_df, calib_df, test_df, max_age = load_data(sample_frac=sample_frac, seed=args.seed)
    print(f"Splits -> train {len(train_df)} | val {len(val_df)} | "
          f"calibration {len(calib_df)} | test {len(test_df)}")
    
    print("Building PyTorch DataLoaders...")
    train_loader, val_loader = build_datasets(train_df, val_df, backbone_name=backbone_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building model...")
    model = build_multi_input_model(name=backbone_name)
    model = model.to(device)
    
    criterion = nn.SmoothL1Loss() # Huber / smooth L1 for training
    mae_metric = nn.L1Loss()      # MAE for monitoring / checkpointing
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_mae = float('inf')
    #early_stop_patience = 5
    #early_stop_counter = 0
    history = {'loss': [], 'val_loss': [], 'val_mae': []}
    
    checkpoint_path = None

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            inputs, targets = batch
            img = inputs['image_input'].to(device)
            sex = inputs['sex_input'].to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(img, sex)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * img.size(0)
            progress_bar.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_loader.dataset)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                img = inputs['image_input'].to(device)
                sex = inputs['sex_input'].to(device)
                targets = targets.to(device)
                
                outputs = model(img, sex)
                loss = mae_metric(outputs, targets)
                val_loss += loss.item() * img.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss (MAE): {val_loss:.4f} - lr: {current_lr:.6f}")
        
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_loss)
        
        scheduler.step(val_loss)
        
        # Model Checkpointing
        if val_loss < best_val_mae:
            new_checkpoint_path = run_dir / f"best_model.pth"
            print(f"val_loss improved from {best_val_mae:.4f} to {val_loss:.4f}, saving model to {new_checkpoint_path}")
            best_val_mae = val_loss
            
            if checkpoint_path and checkpoint_path.exists():
                checkpoint_path.unlink()
                
            checkpoint_path = new_checkpoint_path
            torch.save(model.state_dict(), checkpoint_path)
        #     early_stop_counter = 0
        # else:
        #     early_stop_counter += 1
        #     if early_stop_counter >= early_stop_patience:
        #         print(f"Early stopping triggered after {epoch+1} epochs.")
        #         break
            
    # Load best weights for evaluation
    if checkpoint_path and checkpoint_path.exists():
        print(f"Restoring model weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    history_path = run_dir / f"history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Saved training history to {history_path}")

    # Plot training and validation loss
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(history['loss']) + 1)
        plt.plot(epochs_range, history['loss'], label='Train Loss (Huber)', marker='o', linewidth=2)
        plt.plot(epochs_range, history['val_loss'], label='Val Loss (MAE)', marker='s', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plot_path = run_dir / f"loss_plot.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved loss plot to {plot_path}")
    except ImportError:
        print("matplotlib or seaborn not installed, skipping plot generation.")

    evaluate_and_save_metrics(model, val_loader, val_df, max_age, run_dir, seed=args.seed, split="val", device=device)
    
    print("\nRunning evaluation on test set...")
    test_loader = build_val_or_test_loader(test_df, backbone_name=backbone_name)
    evaluate_and_save_metrics(model, test_loader, test_df, max_age, run_dir, seed=args.seed, split="test", device=device)
    
    print("Training process completed.")

if __name__ == "__main__":
    main()
