#!/usr/bin/env python3
"""Bayesian Neural Network (bayesian-torch) training + UQ inference for bone age.

Trains a BNN (deterministic backbone + Bayesian regression head) with the ELBO
objective ``SmoothL1 + kl_beta/N * KL``, selecting the checkpoint with the lowest
validation MAE. It then draws ``n_passes`` stochastic samples from the posterior
predictive on the held-out test set and reports interval metrics (PICP / MPIW /
PINAW / CWC) at the requested coverage levels.

Run from the project root:
    python bnn/bnn.py --backbone efficientnet_b3 --seed 0 --n-passes 60
"""

import sys
import ast
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
# Project root for shared utilities; bnn/ takes priority so `model` resolves to
# bnn/model.py rather than the root point-prediction model.
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from data_loader import load_data, build_datasets, build_val_or_test_loader
from config import OUTPUT_DIR
from uq_metrics import compute_metrics, coverage_dict

from model import build_bnn_model  # bnn/model.py (same directory)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def _mae_over_passes(model, loader, device, n_passes):
    """Mean predictive MAE (normalized scale) averaged over `n_passes` samples."""
    model.eval()
    total_abs, total_n = 0.0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            img = inputs["image_input"].to(device)
            sex = inputs["sex_input"].to(device)
            targets = targets.to(device)

            preds = torch.zeros_like(targets)
            for _ in range(n_passes):
                preds += model(img, sex)
            preds /= n_passes

            total_abs += torch.abs(preds - targets).sum().item()
            total_n += targets.numel()
    return total_abs / total_n


def train(model, train_loader, val_loader, device, epochs, kl_beta,
          val_passes, run_dir):
    """ELBO training loop; keeps the checkpoint with the best validation MAE."""
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    num_train = len(train_loader.dataset)
    history = {"loss": [], "recon_loss": [], "kl_loss": [], "val_mae": []}
    best_val_mae = float("inf")
    ckpt_path = run_dir / "best_bnn_model.pth"

    for epoch in range(epochs):
        model.train()
        run_loss = run_recon = run_kl = 0.0
        for inputs, targets in train_loader:
            img = inputs["image_input"].to(device)
            sex = inputs["sex_input"].to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(img, sex)
            recon = criterion(outputs, targets)
            kl = model.kl_loss()
            loss = recon + kl_beta * kl / num_train
            loss.backward()
            optimizer.step()

            bs = targets.size(0)
            run_loss += loss.item() * bs
            run_recon += recon.item() * bs
            run_kl += kl.item() * bs

        run_loss /= num_train
        run_recon /= num_train
        run_kl /= num_train

        val_mae = _mae_over_passes(model, val_loader, device, val_passes)
        scheduler.step(val_mae)

        history["loss"].append(run_loss)
        history["recon_loss"].append(run_recon)
        history["kl_loss"].append(run_kl)
        history["val_mae"].append(val_mae)

        marker = ""
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), ckpt_path)
            marker = "  <-- best"
        print(f"  epoch {epoch + 1:3d}/{epochs} | loss={run_loss:.4f} "
              f"recon={run_recon:.4f} kl={run_kl:.1f} | val_mae={val_mae:.4f}{marker}")

    # Restore best weights
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    with open(run_dir / "history.pkl", "wb") as f:
        pickle.dump(history, f)

    return best_val_mae


# -----------------------------------------------------------------------------
# Inference (posterior predictive)
# -----------------------------------------------------------------------------
@torch.no_grad()
def bnn_forward_passes(model, loader, device, n_passes):
    """Draw `n_passes` posterior samples.

    Returns
    -------
    preds   : np.ndarray (n_passes, n_samples) — normalized scale
    targets : np.ndarray (n_samples,)          — normalized scale
    """
    model.eval()
    pass_preds = [[] for _ in range(n_passes)]
    all_targets = []

    for inputs, targets in loader:
        img = inputs["image_input"].to(device)
        sex = inputs["sex_input"].to(device)

        for p in range(n_passes):
            out = model(img, sex).squeeze(1).cpu().numpy()
            pass_preds[p].append(out)

        all_targets.append(targets.squeeze(1).numpy())

    preds = np.stack([np.concatenate(pass_preds[p]) for p in range(n_passes)], axis=0)
    targets = np.concatenate(all_targets)
    return preds, targets


def evaluate(preds, targets, backbone, seed, n_passes, elapsed, coverage, df):
    """Aggregate posterior samples into mean/std, compute + save UQ metrics."""
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)

    mae = mean_absolute_error(targets, means)
    mse = mean_squared_error(targets, means)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, means)

    metrics = {
        "backbone": backbone,
        "seed": seed,
        "n_passes": n_passes,
        "n_samples": len(targets),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "inference_time_s": elapsed,
    }
    for cov in coverage:
        z = coverage_dict[cov]
        lower = means - z * stds
        upper = means + z * stds
        picp, mpiw, pinaw, cwc = compute_metrics(lower, upper, targets, cov)
        metrics[f"picp_{int(cov * 100)}"] = picp
        metrics[f"mpiw_{int(cov * 100)}"] = mpiw
        metrics[f"pinaw_{int(cov * 100)}"] = pinaw
        metrics[f"cwc_{int(cov * 100)}"] = cwc

    out_dir = OUTPUT_DIR / "bnn" / backbone / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_excel(out_dir / "bnn_metrics.xlsx", index=False)

    pred_df = df[["id", "boneage", "male"]].copy().reset_index(drop=True)
    pred_df["pred_mean"] = means
    pred_df["pred_std"] = stds
    for cov in coverage:
        pred_df[f"lower_{int(cov * 100)}"] = means - coverage_dict[cov] * stds
        pred_df[f"upper_{int(cov * 100)}"] = means + coverage_dict[cov] * stds
    pred_df["abs_error"] = np.abs(means - targets)
    pred_df.to_csv(out_dir / "bnn_predictions.csv", index=False)

    print(f"  saved -> {out_dir}")
    for cov in coverage:
        print(f"    coverage {int(cov * 100)}%: "
              f"PICP={metrics[f'picp_{int(cov * 100)}']:.3f} "
              f"PINAW={metrics[f'pinaw_{int(cov * 100)}']:.3f} "
              f"CWC={metrics[f'cwc_{int(cov * 100)}']:.3f}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BNN training + UQ inference — bone age")
    parser.add_argument("--backbone", default="efficientnet_b3",
                        choices=["efficientnet_b3", "vit_b_16", "convnextv2_tiny"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-passes", type=int, default=60,
                        help="Posterior samples drawn at inference for UQ")
    parser.add_argument("--val-passes", type=int, default=5,
                        help="Posterior samples averaged for validation MAE")
    parser.add_argument("--prior-sigma", type=float, default=1.0,
                        help="Std dev of the Gaussian weight prior")
    parser.add_argument("--kl-beta", type=float, default=0.1,
                        help="Weight of the KL term in the ELBO")
    parser.add_argument("--coverage", type=ast.literal_eval, default=[0.90, 0.95, 0.99])
    parser.add_argument("--quick-test", action="store_true",
                        help="1%% of data, 2 epochs, 5 passes — for debugging")
    args = parser.parse_args()

    sample_frac = 0.01 if args.quick_test else 1.0
    epochs = 2 if args.quick_test else args.epochs
    n_passes = 5 if args.quick_test else args.n_passes
    val_passes = 2 if args.quick_test else args.val_passes

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[BNN] backbone={args.backbone} seed={args.seed} "
          f"epochs={epochs} passes={n_passes} device={device}")

    run_dir = OUTPUT_DIR / "bnn" / args.backbone / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_df, val_df, calib_df, test_df, max_age = load_data(
        sample_frac=sample_frac, seed=args.seed)
    train_loader, val_loader = build_datasets(
        train_df, val_df, backbone_name=args.backbone)
    test_loader = build_val_or_test_loader(test_df, backbone_name=args.backbone)
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"| max_age={max_age:.0f} months")

    # Model
    model = build_bnn_model(name=args.backbone, prior_sigma=args.prior_sigma).to(device)

    # Train
    best_val_mae = train(model, train_loader, val_loader, device, epochs,
                         args.kl_beta, val_passes, run_dir)
    print(f"  best val MAE (normalized) = {best_val_mae:.4f} "
          f"({best_val_mae * max_age:.2f} months)")

    # Inference on the held-out test set
    print(f"  running {n_passes} posterior passes on test set...")
    t0 = time.perf_counter()
    preds_norm, targets_norm = bnn_forward_passes(model, test_loader, device, n_passes)
    elapsed = time.perf_counter() - t0

    preds = preds_norm * max_age
    targets = targets_norm * max_age
    evaluate(preds, targets, args.backbone, args.seed, n_passes, elapsed,
             args.coverage, test_df)


if __name__ == "__main__":
    main()
