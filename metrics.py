import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
def evaluate_and_save_metrics(model, val_loader, val_df, max_age, run_dir, seed, device="cpu", split="val"):
    """
    Evaluates the PyTorch model on the given dataset, de-normalizes predictions,
    calculates metrics (MAE, RMSE, MSE, R2, accuracy), and saves them to CSV.
    `split` labels the set (e.g. "val" or "test") in prints and output filenames.
    """
    print(f"\n[{seed}] Evaluating model on {split} set...")
    model.eval()
    
    preds = []
    with torch.no_grad():
        for batch in val_loader:
            inputs, _ = batch
            img = inputs['image_input'].to(device)
            sex = inputs['sex_input'].to(device)
            out = model(img, sex)
            preds.append(out.cpu().numpy())
            
    # Flatten the list of batch predictions
    y_norm_pred = np.vstack(preds).flatten()
    
    # De-normalize
    y_pred = y_norm_pred * max_age
    y_true = val_df['boneage_norm'].values * max_age
    
    # Calculate metrics
    mae   = mean_absolute_error(y_true, y_pred)
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    r2    = r2_score(y_true, y_pred)
    acc6  = np.mean(np.abs(y_pred - y_true) <= 6.0)  * 100
    acc12 = np.mean(np.abs(y_pred - y_true) <= 12.0) * 100

    print(f"\n--- {split.capitalize()} Metrics ---")
    print(f"MAE:             {mae:.2f} months")
    print(f"RMSE:            {rmse:.2f} months")
    print(f"MSE:             {mse:.2f}")
    print(f"R²:              {r2:.3f}")
    print(f"% within ±6 mo:  {acc6:.1f}%")
    print(f"% within ±12 mo: {acc12:.1f}%")
    print("--------------------------\n")

    # Generate timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Predictions DataFrame
    predictions_df = pd.DataFrame({
        "id":        val_df["id"].values,
        "sex":       val_df["male"].values.astype(int),
        "true_age":  y_true,
        "pred_age":  y_pred,
    })
    predictions_df["abs_error"] = (predictions_df["pred_age"] - predictions_df["true_age"]).abs()
    
    pred_path = run_dir / f"{split}_predictions_{ts}.csv"
    predictions_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to: {pred_path} ({len(predictions_df)} rows)")

    # Save Metrics DataFrame
    metrics_df = pd.DataFrame([{
        "seed":           seed,
        "MAE":            mae,
        "RMSE":           rmse,
        "MSE":            mse,
        "R2":             r2,
        "pct_within_6mo": acc6,
        "pct_within_12mo": acc12,
        "n_val":          len(predictions_df),
        "timestamp":      ts,
    }])
    
    metrics_path = run_dir / f"{split}_metrics_{ts}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to:     {metrics_path}")
    
    return metrics_df
