#!/usr/bin/env python3
"""Monte Carlo Dropout inference for bone age uncertainty quantification.

Run from project root:
    python mcd/run_mcd.py --backbone efficientnet_b3 --seed 0
"""

from os import mkdir
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import ast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import build_multi_input_model
from data_loader import load_data, build_val_or_test_loader
from config import OUTPUT_DIR
from uq_metrics import compute_metrics, coverage_dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def enable_dropout(model: nn.Module) -> None:
    """Set all Dropout layers to train mode while the rest stays in eval mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_forward_passes(
    model: nn.Module,
    loader,
    device: torch.device,
    n_passes: int,
):
    """Run n_passes stochastic forward passes.

    Returns
    -------
    preds   : np.ndarray (n_passes, n_samples) — normalized scale
    targets : np.ndarray (n_samples,)           — normalized scale
    """
    model.eval()
    enable_dropout(model)

    pass_preds = [[] for _ in range(n_passes)]
    all_targets = []

    for inputs, targets in loader:
        img = inputs["image_input"].to(device)
        sex = inputs["sex_input"].to(device)

        for p in range(n_passes):
            out = model(img, sex).squeeze(1).cpu().numpy()
            pass_preds[p].append(out)

        all_targets.append(targets.squeeze(1).numpy())

    # Sonuc olarak n_passes x n_samples boyutunda bir matrix elde ediyoruz.
    preds = np.stack(
        [np.concatenate(pass_preds[p]) for p in range(n_passes)], axis=0
    )
    targets = np.concatenate(all_targets)
    return preds, targets

def evaluate(preds, targets, backbone, seed, n_passes, elapsed,coverage,df):
    means = preds.mean(axis=0)
    stds  = preds.std(axis=0)

    # Calculate metrics
    mae            = mean_absolute_error(means, targets)
    mse            = mean_squared_error(means, targets)
    rmse           = np.sqrt(mse)
    r2             = r2_score(means, targets)

    metrics = {
        "backbone":         backbone,
        "seed":             seed,
        "n_passes":         n_passes,
        "n_samples":        len(targets),
        "mae":              mae,
        "mse":              mse,
        "rmse":             rmse,
        "r2":               r2,
        "inference_time_s": elapsed,
    }
    for cov in coverage:
        z_score = coverage_dict[cov]

        lower = means - z_score * stds
        upper = means + z_score * stds  
        picp, mpiw, pinaw, cwc = compute_metrics(lower, upper, targets, cov)
        metrics[f"picp_{int(cov*100)}"] = picp
        metrics[f"mpiw_{int(cov*100)}"] = mpiw
        metrics[f"pinaw_{int(cov*100)}"] = pinaw
        metrics[f"cwc_{int(cov*100)}"] = cwc
    
    out_dir = OUTPUT_DIR / "mcd" / backbone / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    excel_path = out_dir / f"mcd_metrics.xlsx"
    pd.DataFrame([metrics]).to_excel(excel_path, index=False)

    pred_df = df[["id", "boneage", "male"]].copy().reset_index(drop=True)
    pred_df["pred_mean"] = means
    pred_df["pred_std"]  = stds
    for cov in coverage:
        pred_df[f"lower_{int(cov*100)}"]  = means - coverage_dict[cov] * stds
        pred_df[f"upper_{int(cov*100)}"]  = means + coverage_dict[cov] * stds
    pred_df["abs_error"] = np.abs(means - targets)
    pred_df.to_csv(out_dir / f"mcd_predictions.csv", index=False)

def run_mcd(backbone: str, seed: int, n_passes: int) -> dict:
    checkpoint = OUTPUT_DIR / backbone / f"seed_{seed:02d}" / "model_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    _, val_df, calib_df, test_df, max_age = load_data(seed=seed)

    loader = build_val_or_test_loader(test_df, backbone_name=backbone)
    print(f"{len(test_df)} samples | max_age={max_age:.0f} months")

    model = build_multi_input_model(name=backbone)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    print(f"  Running {n_passes} MC passes...")
    t0 = time.perf_counter()
    preds_norm, targets_norm = mc_forward_passes(model, loader, device, n_passes)
    elapsed = time.perf_counter() - t0

    preds   = preds_norm   * max_age   # (n_passes, n_samples) in months
    targets = targets_norm * max_age   # (n_samples,) in months

    return preds, targets, test_df, elapsed


def main():
    parser = argparse.ArgumentParser(description="MC Dropout inference — bone age uncertainty")
    parser.add_argument("--backbone", default="efficientnet_b3",
                        choices=["efficientnet_b3", "vit_b_16", "convnextv2_tiny"])
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--n-passes",  type=int, default=30)
    parser.add_argument("--conformalized", type=int, default=0)
    parser.add_argument("--quick-test", action="store_true",
                        help="Use 5 MC passes for fast debugging")
    parser.add_argument("--coverage", type=ast.literal_eval, default=[0.90,0.95,0.99])
    args = parser.parse_args()

    n_passes = 5 if args.quick_test else args.n_passes

    print(f"\n[MCD] backbone={args.backbone}  seed={args.seed}  passes={n_passes}")

    
    preds, targets, df, elapsed = run_mcd(args.backbone, args.seed, n_passes)
    evaluate(preds, targets, args.backbone, args.seed, n_passes, elapsed,args.coverage, df)


if __name__ == "__main__":
    main()
