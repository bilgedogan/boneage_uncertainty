import os
import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import BASE_DIR
from data_loader import build_datasets, build_val_or_test_loader, load_data
from cqr.conformal import calibrate_cqr, evaluate_and_save_cqr_metrics
from cqr.losses import build_quantile_loss
from cqr.model import build_cqr_model


QUICK_TEST = os.getenv("CQR_QUICK_TEST", "0") == "1"
EPOCHS = int(os.getenv("CQR_EPOCHS", "50"))
SEED = int(os.getenv("CQR_SEED", "0"))
BACKBONE = os.getenv("CQR_BACKBONE", "efficientnet_b3")
ALPHA = float(os.getenv("CQR_ALPHA", "0.1"))
LEARNING_RATE = float(os.getenv("CQR_LEARNING_RATE", "1e-4"))
DROPOUT = float(os.getenv("CQR_DROPOUT", "0.5"))
LAMBDA_CROSS = float(os.getenv("CQR_LAMBDA_CROSS", "10.0"))
CROSSING_MARGIN = float(os.getenv("CQR_CROSSING_MARGIN", "0.0"))
CROSSING_POWER = int(os.getenv("CQR_CROSSING_POWER", "1"))
EPOCH_LOG_INTERVAL = int(os.getenv("CQR_EPOCH_LOG_INTERVAL", "5"))
OUTPUT_DIR = Path(os.getenv("CQR_OUTPUT_DIR", BASE_DIR / "cqr" / "outputs"))
SEEDS = [int(seed) for seed in os.getenv("CQR_SEEDS", "0 1 2 3 4 5 6 7 8 9").split()]


def _center_mae(preds, targets):
    center = preds.mean(dim=1, keepdim=True)
    return nn.functional.l1_loss(center, targets)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(loader, desc="Training")
    for inputs, targets in progress_bar:
        img = inputs["image_input"].to(device)
        sex = inputs["sex_input"].to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(img, sex)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)
        progress_bar.set_postfix({"loss": loss.item()})

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    loss_total = 0.0
    mae_total = 0.0

    with torch.no_grad():
        for inputs, targets in loader:
            img = inputs["image_input"].to(device)
            sex = inputs["sex_input"].to(device)
            targets = targets.to(device)

            preds = model(img, sex)
            loss_total += criterion(preds, targets).item() * img.size(0)
            mae_total += _center_mae(preds, targets).item() * img.size(0)

    n = len(loader.dataset)
    return loss_total / n, mae_total / n


def should_log_epoch(epoch, epochs):
    return epoch == 0 or (epoch + 1) % EPOCH_LOG_INTERVAL == 0 or epoch == epochs - 1


def print_metrics(split, metrics_df):
    row = metrics_df.iloc[0]
    print(
        f"{split}: picp={row['picp']:.3f} pinaw={row['pinaw']:.3f} "
        f"cwc={row['cwc']:.3f} width={row['mean_width']:.2f}mo "
        f"center_MAE={row['center_MAE']:.2f}mo"
    )


def run_name(seed):
    name = f"cqr_alpha{ALPHA:g}_seed{seed}"
    return name + ("_quicktest" if QUICK_TEST else "")


def run_dir(seed):
    return OUTPUT_DIR / BACKBONE / run_name(seed)


def summarize(per_seed):
    numeric = per_seed.select_dtypes(include="number").drop(columns=["seed"], errors="ignore")
    return pd.DataFrame([
        {"metric": col, "mean": numeric[col].mean(), "std": numeric[col].std(ddof=1)}
        for col in numeric.columns
    ])


def aggregate_results():
    summary_dir = OUTPUT_DIR / BACKBONE / (
        f"cqr_alpha{ALPHA:g}_summary"
        + ("_quicktest" if QUICK_TEST else "")
    )
    summary_dir.mkdir(parents=True, exist_ok=True)

    for split in ["val", "test"]:
        rows = []
        for seed in SEEDS:
            metrics_path = run_dir(seed) / f"{split}_cqr_metrics.csv"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics for seed={seed}: {metrics_path}")
            row = pd.read_csv(metrics_path)
            row["run_dir"] = str(metrics_path.parent)
            rows.append(row)

        per_seed = pd.concat(rows, ignore_index=True)
        per_seed.to_csv(summary_dir / f"{split}_per_seed_metrics.csv", index=False)
        summarize(per_seed).to_csv(summary_dir / f"{split}_mean_std_metrics.csv", index=False)
        print(f"{split}: summary saved to {summary_dir}")


def main():
    if os.getenv("CQR_AGGREGATE", "0") == "1":
        aggregate_results()
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CQR experiments require CUDA. No GPU was detected.")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    sample_frac = 0.01 if QUICK_TEST else 1.0
    epochs = 2 if QUICK_TEST else EPOCHS
    current_run_name = run_name(SEED)
    current_run_dir = run_dir(SEED)
    current_run_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, calib_df, test_df, max_age = load_data(sample_frac=sample_frac, seed=SEED)
    print(
        f"CQR run={current_run_name} backbone={BACKBONE} epochs={epochs} "
        f"train={len(train_df)} val={len(val_df)} calib={len(calib_df)} test={len(test_df)}"
    )

    train_loader, val_loader = build_datasets(train_df, val_df, backbone_name=BACKBONE)
    calib_loader = build_val_or_test_loader(calib_df, backbone_name=BACKBONE)
    test_loader = build_val_or_test_loader(test_df, backbone_name=BACKBONE)

    device = torch.device("cuda")
    print(f"Using device: {device}")

    model = build_cqr_model(dropout=DROPOUT, name=BACKBONE).to(device)
    criterion = build_quantile_loss(
        alpha=ALPHA,
        lambda_cross=LAMBDA_CROSS,
        margin=CROSSING_MARGIN,
        crossing_power=CROSSING_POWER,
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    checkpoint_path = current_run_dir / "best_cqr_model.pth"
    history = {"loss": [], "val_loss": [], "val_center_mae": []}

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_center_mae = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        if should_log_epoch(epoch, epochs):
            print(
                f"epoch {epoch + 1}/{epochs}: loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_center_mae={val_center_mae:.4f} "
                f"lr={current_lr:.6f}"
            )

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_center_mae"].append(val_center_mae)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    history_path = current_run_dir / "history.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    q_hat = calibrate_cqr(model, calib_loader, alpha=ALPHA, device=device)
    q_hat_value = float(q_hat.detach().cpu())
    print(f"calibration q_hat={q_hat_value:.6f} ({q_hat_value * max_age:.2f} months)")

    val_metrics = evaluate_and_save_cqr_metrics(
        model,
        val_loader,
        val_df,
        max_age,
        current_run_dir,
        seed=SEED,
        q_hat=q_hat,
        alpha=ALPHA,
        split="val",
        device=device,
    )
    test_metrics = evaluate_and_save_cqr_metrics(
        model,
        test_loader,
        test_df,
        max_age,
        current_run_dir,
        seed=SEED,
        q_hat=q_hat,
        alpha=ALPHA,
        split="test",
        device=device,
    )

    print_metrics("val", val_metrics)
    print_metrics("test", test_metrics)
    print(f"outputs={current_run_dir}")


if __name__ == "__main__":
    main()
