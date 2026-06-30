import argparse
import ast
import pickle
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add root dir to path so we can import shared modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Set DATA_DIR environment variable to ../data as required
import os
if "DATA_DIR" not in os.environ:
    os.environ["DATA_DIR"] = "../data"

# Now we can import root modules
from config import OUTPUT_DIR
from data_loader import load_data, build_datasets, build_val_or_test_loader
from metrics import evaluate_and_save_metrics
from uq_metrics import compute_metrics, coverage_dict

# Import our BNN model
from model import build_multi_input_model

class PointwiseModelWrapper(nn.Module):
    """Wraps the BNN to always use mean weights (sample=False) for pointwise metrics."""
    def __init__(self, bnn_model):
        super().__init__()
        self.bnn_model = bnn_model
        
    def forward(self, img, sex):
        return self.bnn_model(img, sex, sample=False)

def evaluate_uq(
    model,
    loader,
    df,
    max_age,
    run_dir,
    split="val",
    device="cpu",
    n_passes=30,
    coverage_levels=None,
    backbone=None,
    seed=None,
):
    if coverage_levels is None:
        coverage_levels = [0.90, 0.95, 0.99]

    print(f"\nEvaluating UQ on {split} set with {n_passes} passes...")
    model.eval()

    all_batch_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"UQ {split}"):
            inputs, _ = batch
            img = inputs['image_input'].to(device)
            sex = inputs['sex_input'].to(device)

            batch_preds = []
            for _ in range(n_passes):
                out = model(img, sex, sample=True)
                batch_preds.append(out.cpu().numpy())

            stacked_preds = np.stack(batch_preds, axis=0)
            all_batch_preds.append(stacked_preds)

    all_preds = np.concatenate(all_batch_preds, axis=1)
    all_preds = all_preds.squeeze(-1)

    all_preds = all_preds * max_age
    y_true = df['boneage_norm'].values * max_age

    pred_mean = all_preds.mean(axis=0)
    pred_std = all_preds.std(axis=0)

    metrics = {
        "backbone": backbone,
        "seed": seed,
        "split": split,
        "n_passes": n_passes,
    }

    print(f"\n--- {split.capitalize()} UQ Metrics ---")
    for cov in coverage_levels:
        z_score = coverage_dict[cov]
        lower = pred_mean - z_score * pred_std
        upper = pred_mean + z_score * pred_std
        picp, mpiw, pinaw, cwc = compute_metrics(lower, upper, y_true, cov)
        cov_key = int(cov * 100)
        metrics[f"picp_{cov_key}"] = picp
        metrics[f"mpiw_{cov_key}"] = mpiw
        metrics[f"pinaw_{cov_key}"] = pinaw
        metrics[f"cwc_{cov_key}"] = cwc
        print(f"  [{cov_key}%] PICP: {picp:.3f}  MPIW: {mpiw:.3f}  PINAW: {pinaw:.3f}  CWC: {cwc:.3f}")
    print("--------------------------\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics["timestamp"] = ts

    uq_preds_df = pd.DataFrame({
        "id": df["id"].values,
        "true_age": y_true,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
    })
    for cov in coverage_levels:
        cov_key = int(cov * 100)
        z_score = coverage_dict[cov]
        uq_preds_df[f"lower_{cov_key}"] = pred_mean - z_score * pred_std
        uq_preds_df[f"upper_{cov_key}"] = pred_mean + z_score * pred_std

    pred_path = run_dir / f"{split}_uq_predictions_{ts}.csv"
    uq_preds_df.to_csv(pred_path, index=False)
    print(f"Saved UQ predictions to: {pred_path}")

    metrics_df = pd.DataFrame([metrics])
    metrics_path = run_dir / f"{split}_uq_metrics_{ts}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved UQ metrics to: {metrics_path}")

    seed_dir = run_dir.parent
    stable_metrics_path = seed_dir / f"bnn_uq_metrics_{split}.csv"
    metrics_df.to_csv(stable_metrics_path, index=False)
    print(f"Saved stable UQ metrics to: {stable_metrics_path}")

    return metrics_df

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Input BNN Model with PyTorch")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick test with 1% of data and 2 epochs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--seed", type=int, default=0, help="Torch init/training seed (data split stays fixed)")
    parser.add_argument("--backbone", type=str, default="efficientnet_b3", help="Backbone to use")
    parser.add_argument("--n-passes", type=int, default=30, help="Number of forward passes for UQ evaluation")
    parser.add_argument("--prior-sigma", type=float, default=1.0, help="Prior std dev for BNN weights")
    parser.add_argument("--coverage", type=ast.literal_eval, default=[0.90, 0.95, 0.99])
    args = parser.parse_args()

    # Seed torch only -> isolates model-init/training stochasticity.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sample_frac = 0.01 if args.quick_test else 1.0
    epochs = 2 if args.quick_test else args.epochs
    backbone_name = args.backbone
    run_name = f"seed_{args.seed:02d}" + ("_quicktest" if args.quick_test else "")

    # Output directory for bnn
    bnn_output_dir = PROJECT_ROOT / "bnn" / "results"
    run_dir = bnn_output_dir / backbone_name / run_name
    
    # Add unique timestamp for the run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BNN - Backbone: {backbone_name}] Starting run in: {run_dir}")
    print(f"Using {sample_frac*100}% of data for {epochs} epochs.")

    print("Loading data...")
    train_df, val_df, calib_df, test_df, max_age = load_data(sample_frac=sample_frac, seed=args.seed)
    
    print("Building PyTorch DataLoaders...")
    train_loader, val_loader = build_datasets(train_df, val_df, backbone_name=backbone_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building model...")
    model = build_multi_input_model(name=backbone_name, prior_sigma=args.prior_sigma)
    model = model.to(device)
    
    criterion = nn.SmoothL1Loss()
    mae_metric = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_mae = float('inf')
    history = {'loss': [], 'recon_loss': [], 'kl_loss': [], 'val_loss': [], 'val_mae': []}
    
    checkpoint_path = None
    num_train_samples = len(train_loader.dataset)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            inputs, targets = batch
            img = inputs['image_input'].to(device)
            sex = inputs['sex_input'].to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with sampling
            outputs = model(img, sex, sample=True)
            
            # Losses
            recon_loss = criterion(outputs, targets)
            kl_div = model.kl_divergence()
            
            # Total loss (KL is scaled by 1/N)
            kl_weight = 1.0 / num_train_samples
            loss = recon_loss + kl_div * kl_weight
            
            loss.backward()
            optimizer.step()
            
            batch_size = img.size(0)
            train_loss += loss.item() * batch_size
            train_recon += recon_loss.item() * batch_size
            train_kl += kl_div.item() * batch_size
            
            progress_bar.set_postfix({
                'loss': loss.item(), 
                'recon': recon_loss.item(), 
                'kl': (kl_div.item() * kl_weight)
            })
            
        train_loss /= num_train_samples
        train_recon /= num_train_samples
        train_kl /= num_train_samples
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                img = inputs['image_input'].to(device)
                sex = inputs['sex_input'].to(device)
                targets = targets.to(device)
                
                # Use mean weights for validation monitoring
                outputs = model(img, sex, sample=False)
                loss = mae_metric(outputs, targets)
                val_loss += loss.item() * img.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl/num_train_samples:.4f}) - val_loss (MAE): {val_loss:.4f} - lr: {current_lr:.6f}")
        
        history['loss'].append(train_loss)
        history['recon_loss'].append(train_recon)
        history['kl_loss'].append(train_kl / num_train_samples)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_mae:
            new_checkpoint_path = run_dir / f"best_model.pth"
            print(f"val_loss improved from {best_val_mae:.4f} to {val_loss:.4f}, saving model to {new_checkpoint_path}")
            best_val_mae = val_loss
            
            if checkpoint_path and checkpoint_path.exists():
                checkpoint_path.unlink()
                
            checkpoint_path = new_checkpoint_path
            torch.save(model.state_dict(), checkpoint_path)
            
    if checkpoint_path and checkpoint_path.exists():
        print(f"Restoring model weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    history_path = run_dir / f"history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(history['loss']) + 1)
        plt.plot(epochs_range, history['loss'], label='Train Total Loss')
        plt.plot(epochs_range, history['recon_loss'], label='Train Recon Loss')
        plt.plot(epochs_range, history['val_loss'], label='Val Loss (MAE)')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plot_path = run_dir / f"loss_plot.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
    except ImportError:
        pass

    # Pointwise evaluation
    print("\nRunning Pointwise Evaluation...")
    pw_wrapper = PointwiseModelWrapper(model)
    evaluate_and_save_metrics(pw_wrapper, val_loader, val_df, max_age, run_dir, seed=args.seed, split="val", device=device)
    
    print("\nRunning UQ Evaluation...")
    evaluate_uq(
        model, val_loader, val_df, max_age, run_dir,
        split="val", device=device, n_passes=args.n_passes,
        coverage_levels=args.coverage, backbone=backbone_name, seed=args.seed,
    )

    # Test set evaluation
    print("\nRunning evaluation on test set...")
    test_loader = build_val_or_test_loader(test_df, backbone_name=backbone_name)
    evaluate_and_save_metrics(pw_wrapper, test_loader, test_df, max_age, run_dir, seed=args.seed, split="test", device=device)
    evaluate_uq(
        model, test_loader, test_df, max_age, run_dir,
        split="test", device=device, n_passes=args.n_passes,
        coverage_levels=args.coverage, backbone=backbone_name, seed=args.seed,
    )
    
    print("Training process completed.")

if __name__ == "__main__":
    main()
