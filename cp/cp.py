#!/usr/bin/env python3
"""Split / kNN conformal prediction for bone age uncertainty quantification.

Run from project root:
    python cp/cp.py --backbone efficientnet_b3 --seed 0
"""

import sys
import time
import argparse
import ast
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import build_multi_input_model
from data_loader import load_data, build_val_or_test_loader
from config import OUTPUT_DIR
from uq_metrics import compute_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from crepes import ConformalRegressor
from crepes.extras import DifficultyEstimator


@torch.no_grad()
def point_predict(model, loader, device):
    """Single deterministic forward pass.

    Also returns the fc1-input embedding (backbone features concatenated
    with the processed sex feature), used as the object representation for
    the kNN difficulty estimator.

    Returns
    -------
    preds      : np.ndarray (n_samples,)      — normalized scale
    targets    : np.ndarray (n_samples,)      — normalized scale
    embeddings : np.ndarray (n_samples, dim)
    """
    model.eval()

    all_preds, all_targets, all_emb = [], [], []

    for inputs, targets in loader:
        img = inputs["image_input"].to(device)
        sex = inputs["sex_input"].to(device)

        feat = model.base_model(img)
        sex_proc = model.relu(model.sex_fc(sex))
        emb = torch.cat((feat, sex_proc), dim=1)
        x = model.dropout(model.relu(model.fc1(emb)))
        out = model.out(x).squeeze(1)

        all_preds.append(out.cpu().numpy())
        all_emb.append(emb.cpu().numpy())
        all_targets.append(targets.squeeze(1).numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    embeddings = np.concatenate(all_emb, axis=0)
    return preds, targets, embeddings


def conformalize_split(calib_preds, calib_targets, test_preds, coverage_levels):
    """Standard (non-normalized) split conformal prediction."""
    residuals_calib = calib_targets - calib_preds

    cr = ConformalRegressor()
    cr.fit(residuals_calib)

    return {cov: cr.predict_int(test_preds, confidence=cov) for cov in coverage_levels}


def conformalize_knn(calib_preds, calib_targets, calib_emb, test_preds, test_emb,
                      coverage_levels, k):
    """Normalized split conformal prediction with kNN difficulty estimates.

    The difficulty (sigma) of an object is the mean absolute residual of its
    k nearest neighbors in embedding space, so intervals widen in regions of
    feature space where the model is locally less accurate.
    """
    residuals_calib = calib_targets - calib_preds

    de = DifficultyEstimator()
    de.fit(X=calib_emb, residuals=np.abs(residuals_calib), k=k, scaler=True)
    sigmas_calib = de.apply(calib_emb)
    sigmas_test = de.apply(test_emb)

    cr = ConformalRegressor()
    cr.fit(residuals_calib, sigmas=sigmas_calib)

    return {
        cov: cr.predict_int(test_preds, sigmas=sigmas_test, confidence=cov)
        for cov in coverage_levels
    }


def conformalize_rf(
    calib_preds,
    calib_targets,
    calib_emb,
    test_preds,
    test_emb,
    coverage_levels,
    n_estimators,
    min_samples_leaf,
    random_state,
):
    """Normalized split conformal prediction with RF difficulty estimates."""
    residuals_calib = calib_targets - calib_preds
    abs_residuals = np.abs(residuals_calib)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(calib_emb, abs_residuals)

    sigmas_calib = np.maximum(rf.predict(calib_emb), 1e-6)
    sigmas_test = np.maximum(rf.predict(test_emb), 1e-6)

    cr = ConformalRegressor()
    cr.fit(residuals_calib, sigmas=sigmas_calib)

    return {
        cov: cr.predict_int(test_preds, sigmas=sigmas_test, confidence=cov)
        for cov in coverage_levels
    }


def evaluate(backbone, seed, n_calib, elapsed, df, test_preds, test_targets,
             methods_intervals, coverage_levels):
    pred_df = df[["id", "boneage", "male"]].copy().reset_index(drop=True)
    pred_df["pred"] = test_preds

    metrics_rows = []
    for method_name, intervals in methods_intervals.items():
        mae = mean_absolute_error(test_preds, test_targets)
        mse = mean_squared_error(test_preds, test_targets)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_targets, test_preds)

        metrics = {
            "backbone":         backbone,
            "seed":             seed,
            "method":           method_name,
            "n_calib":          n_calib,
            "n_samples":        len(test_targets),
            "mae":              mae,
            "mse":              mse,
            "rmse":             rmse,
            "r2":               r2,
            "inference_time_s": elapsed,
        }
        for cov in coverage_levels:
            lower, upper = intervals[cov][:, 0], intervals[cov][:, 1]
            picp, mpiw, pinaw, cwc = compute_metrics(lower, upper, test_targets, cov)
            metrics[f"picp_{int(cov*100)}"] = picp
            metrics[f"mpiw_{int(cov*100)}"] = mpiw
            metrics[f"pinaw_{int(cov*100)}"] = pinaw
            metrics[f"cwc_{int(cov*100)}"] = cwc

            pred_df[f"lower_{int(cov*100)}_{method_name}"] = lower
            pred_df[f"upper_{int(cov*100)}_{method_name}"] = upper

        metrics_rows.append(metrics)

    pred_df["abs_error"] = np.abs(test_preds - test_targets)

    out_dir = OUTPUT_DIR / "cp" / backbone / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    excel_path = out_dir / "cp_metrics.xlsx"
    pd.DataFrame(metrics_rows).to_excel(excel_path, index=False)

    pred_df.to_csv(out_dir / "cp_predictions.csv", index=False)


def run_cp(backbone: str, seed: int):
    checkpoint = OUTPUT_DIR / backbone / f"seed_{seed:02d}" / "model_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    _, val_df, calib_df, test_df, max_age = load_data(seed=seed)

    calib_loader = build_val_or_test_loader(calib_df, backbone_name=backbone)
    test_loader = build_val_or_test_loader(test_df, backbone_name=backbone)
    print(f"{len(calib_df)} calib | {len(test_df)} test | max_age={max_age:.0f} months")

    model = build_multi_input_model(name=backbone)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)

    print("  Running point-prediction forward passes...")
    t0 = time.perf_counter()
    calib_preds_norm, calib_targets_norm, calib_emb = point_predict(model, calib_loader, device)
    test_preds_norm, test_targets_norm, test_emb = point_predict(model, test_loader, device)
    elapsed = time.perf_counter() - t0

    calib_preds, calib_targets = calib_preds_norm * max_age, calib_targets_norm * max_age
    test_preds, test_targets = test_preds_norm * max_age, test_targets_norm * max_age

    return (calib_preds, calib_targets, calib_emb,
            test_preds, test_targets, test_emb, test_df, elapsed)


def main():
    parser = argparse.ArgumentParser(description="Conformal Prediction — bone age uncertainty")
    parser.add_argument("--backbone", default="efficientnet_b3",
                        choices=["efficientnet_b3", "vit_b_16", "convnextv2_tiny"])
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--knn-k",     type=int, default=25)
    parser.add_argument("--rf-trees",  type=int, default=200)
    parser.add_argument("--rf-min-leaf", type=int, default=5)
    parser.add_argument("--coverage",  type=ast.literal_eval, default=[0.90, 0.95, 0.99])
    args = parser.parse_args()

    print(
        f"\n[CP] backbone={args.backbone} seed={args.seed} "
        f"knn_k={args.knn_k} rf_trees={args.rf_trees}"
    )

    (calib_preds, calib_targets, calib_emb,
     test_preds, test_targets, test_emb, df, elapsed) = run_cp(args.backbone, args.seed)

    split_intervals = conformalize_split(calib_preds, calib_targets, test_preds, args.coverage)
    knn_intervals = conformalize_knn(calib_preds, calib_targets, calib_emb, test_preds, test_emb,
                                      args.coverage, args.knn_k)
    rf_intervals = conformalize_rf(
        calib_preds,
        calib_targets,
        calib_emb,
        test_preds,
        test_emb,
        args.coverage,
        args.rf_trees,
        args.rf_min_leaf,
        args.seed,
    )

    evaluate(
        args.backbone, args.seed, len(calib_targets), elapsed, df,
        test_preds, test_targets,
        {"split": split_intervals, "knn": knn_intervals, "rf": rf_intervals},
        args.coverage,
    )


if __name__ == "__main__":
    main()
