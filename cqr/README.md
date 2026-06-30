# Conformalized Quantile Regression

This package adds a CQR baseline without changing the existing point-prediction
training code.

Run one seed with the constants at the top of `train_cqr.py`:

```bash
python -m cqr.train_cqr
```

Run one backbone for all confidence levels (`0.90`, `0.95`, `0.99`) and all
10 seeds:

```bash
./run_cqr.sh --backbone efficientnet_b3
```

Other backbones:

```bash
./run_cqr.sh --backbone vit_b_16
./run_cqr.sh --backbone convnextv2_tiny
```

The shell script owns the confidence and seed loops. `train_cqr.py` trains one
seed/confidence pair, and the same file aggregates results when
`CQR_AGGREGATE=1`.

Quick test:

```bash
./run_cqr.sh --quick-test
```

The model reuses the existing point-prediction model and replaces only the final
linear layer, changing the output from one scalar to two quantiles:
`alpha / 2` and `1 - alpha / 2`.

Training uses TorchCP's `QuantileLoss` plus a non-crossing penalty:

```text
loss = TorchCP QuantileLoss
       + lambda_cross * mean(ReLU(q_low - q_high + margin))
```

In code this is:

```python
criterion = NonCrossingQuantileLoss(
    alpha=0.10,
    lambda_cross=10.0,
    margin=0.0,
    crossing_power=1,
)
```

The default regularization settings are:

```text
lambda_cross = 10.0
margin = 0.0
crossing_power = 1
```

Calibration uses TorchCP's CQR score:

```text
max(lower - y, y - upper)
```

Outputs are written under:

```text
cqr/outputs/<backbone>/cqr_alpha<alpha>_seed<seed>/
```

Each seed directory stores:

```text
best_cqr_model.pth
history.pkl
val_cqr_predictions.csv
val_cqr_metrics.csv
test_cqr_predictions.csv
test_cqr_metrics.csv
```

The multi-seed summary is written under:

```text
cqr/outputs/<backbone>/cqr_alpha<alpha>_summary/
```

Summary files:

```text
val_per_seed_metrics.csv
val_mean_std_metrics.csv
test_per_seed_metrics.csv
test_mean_std_metrics.csv
```

Metrics include `picp`, `pinaw`, `cwc`, `mpiw`, `mean_width`, `center_MAE`,
`q_hat_months`, and crossing-rate diagnostics.
