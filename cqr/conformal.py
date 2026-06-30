import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cqr.torchcp_regression import CQR
from uq_metrics import CWC_ETA


def predict_quantiles(model, loader, device="cpu"):
    model.eval()
    preds = []

    with torch.no_grad():
        for inputs, _ in loader:
            img = inputs["image_input"].to(device)
            sex = inputs["sex_input"].to(device)
            out = model(img, sex)
            preds.append(out)

    return torch.cat(preds, dim=0)


def cqr_scores(intervals, targets):
    return CQR()(intervals, targets.view(-1, 1)).view(-1)


def conformal_quantile(scores, alpha=0.1):
    if scores.numel() == 0:
        raise ValueError("Calibration scores cannot be empty.")

    n = scores.numel()
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    sorted_scores = torch.sort(scores.view(-1)).values
    return sorted_scores[k - 1]


def calibrate_cqr(model, loader, alpha=0.1, device="cpu"):
    intervals = predict_quantiles(model, loader, device=device)
    targets = torch.cat([target.to(device) for _, target in loader], dim=0)
    scores = cqr_scores(intervals, targets)
    return conformal_quantile(scores, alpha=alpha)


def conformalize_intervals(intervals, q_hat):
    return CQR().generate_intervals(intervals, q_hat.view(1)).squeeze(1)


def interval_metrics(lower, upper, y_true, target_coverage):
    covered = (y_true >= lower) & (y_true <= upper)
    picp = float(np.mean(covered))
    mpiw = float(np.mean(upper - lower))
    y_range = float(np.max(y_true) - np.min(y_true))
    pinaw = mpiw / y_range if y_range > 0 else float("nan")
    gamma = 1.0 if picp < target_coverage else 0.0
    cwc = pinaw * (1.0 + gamma * np.exp(-CWC_ETA * (picp - target_coverage)))
    return picp, mpiw, pinaw, cwc


def evaluate_and_save_cqr_metrics(
    model,
    loader,
    df,
    max_age,
    run_dir,
    seed,
    q_hat,
    alpha=0.1,
    split="test",
    device="cpu",
):
    raw_norm = predict_quantiles(model, loader, device=device)
    conf_norm = conformalize_intervals(raw_norm, q_hat)

    y_true = df["boneage_norm"].values * max_age
    raw_lower = raw_norm[:, 0].detach().cpu().numpy() * max_age
    raw_upper = raw_norm[:, 1].detach().cpu().numpy() * max_age
    lower = conf_norm[:, 0].detach().cpu().numpy() * max_age
    upper = conf_norm[:, 1].detach().cpu().numpy() * max_age
    center = ((lower + upper) / 2.0)

    coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    width = np.mean(upper - lower)
    picp, mpiw, pinaw, cwc = interval_metrics(lower, upper, y_true, 1.0 - alpha)
    raw_crossing_rate = float(np.mean(raw_lower > raw_upper))
    conformal_crossing_rate = float(np.mean(lower > upper))
    mae = mean_absolute_error(y_true, center)
    mse = mean_squared_error(y_true, center)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, center)

    predictions_df = pd.DataFrame({
        "id": df["id"].values,
        "sex": df["male"].values.astype(int),
        "true_age": y_true,
        "raw_lower_age": raw_lower,
        "raw_upper_age": raw_upper,
        "lower_age": lower,
        "upper_age": upper,
        "center_age": center,
    })
    predictions_df["covered"] = (
        (predictions_df["true_age"] >= predictions_df["lower_age"])
        & (predictions_df["true_age"] <= predictions_df["upper_age"])
    )
    predictions_df["interval_width"] = predictions_df["upper_age"] - predictions_df["lower_age"]

    predictions_df.to_csv(run_dir / f"{split}_cqr_predictions.csv", index=False)

    metrics_df = pd.DataFrame([{
        "seed": seed,
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "coverage_percent": coverage,
        "picp": picp,
        "mean_width": width,
        "mpiw": mpiw,
        "pinaw": pinaw,
        "cwc": cwc,
        "center_MAE": mae,
        "center_RMSE": rmse,
        "center_MSE": mse,
        "center_R2": r2,
        "raw_crossing_rate": raw_crossing_rate,
        "conformal_crossing_rate": conformal_crossing_rate,
        "q_hat_norm": float(q_hat.detach().cpu()),
        "q_hat_months": float(q_hat.detach().cpu() * max_age),
        "n": len(predictions_df),
    }])

    metrics_df.to_csv(run_dir / f"{split}_cqr_metrics.csv", index=False)

    return metrics_df
