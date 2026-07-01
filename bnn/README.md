# Bayesian Neural Network (BNN) — bone age UQ

Uncertainty quantification via a Bayesian Neural Network built with Intel Labs'
[`bayesian-torch`](https://github.com/IntelLabs/bayesian-torch).

## Design

- **Backbone** (`efficientnet_b3` / `vit_b_16` / `convnextv2_tiny`) is a
  deterministic, fine-tuned feature extractor — mirrors the point model in
  `../model.py`.
- **Regression head** (`fc1` → ReLU → Dropout → `out`) is converted to Bayesian
  mean-field **Reparameterization** variational layers via `dnn_to_bnn`.
- **Objective**: ELBO = `SmoothL1(pred, target) + kl_beta/N · KL`, where `KL` is
  `get_kl_loss(head)` and `N` is the training-set size.
- **Model selection**: lowest validation MAE (averaged over a few posterior samples).
- **Inference**: `n_passes` posterior samples → per-sample mean/std → Gaussian
  intervals `mean ± z·std` → PICP / MPIW / PINAW / CWC at 90/95/99% coverage.

Shared project utilities are reused as-is: `data_loader.py`, `config.py`,
`uq_metrics.py`.

## Files

| File | Purpose |
|------|---------|
| `model.py` | `build_bnn_model(name, dropout, prior_sigma)` — backbone + Bayesian head |
| `bnn.py`   | Train one (backbone, seed), then run posterior inference + save metrics |
| `run_bnn.sh` | Loop seeds 0–9 for one backbone on a chosen GPU |

## Usage

Requires the `uq` conda env with `bayesian-torch` and `openpyxl` installed, and a
`.env` (see `../.envexample`) defining `DATA_DIR` / `OUTPUT_DIR`.

```bash
# All seeds for one backbone on GPU 0
bash bnn/run_bnn.sh --backbone efficientnet_b3 --n-passes 60 --gpu 0

# Other backbones / GPUs
bash bnn/run_bnn.sh --backbone vit_b_16        --n-passes 60 --gpu 1
bash bnn/run_bnn.sh --backbone convnextv2_tiny --n-passes 60 --gpu 2

# Fast debug (1% data, 2 epochs, 5 passes)
bash bnn/run_bnn.sh --backbone efficientnet_b3 --quick-test --gpu 0

# Single run
python bnn/bnn.py --backbone efficientnet_b3 --seed 0 --n-passes 60
```

`run_bnn.sh` flags: `--backbone`, `--n-passes`, `--coverage`, `--gpu`,
`--quick-test`. Extra knobs (`--epochs`, `--prior-sigma`, `--kl-beta`,
`--val-passes`) are available on `bnn.py` directly.

## Output

Per run → `OUTPUT_DIR/bnn/<backbone>/seed<seed>/`:

- `best_bnn_model.pth` — best-MAE checkpoint
- `history.pkl` — per-epoch loss / recon / kl / val_mae
- `bnn_metrics.xlsx` — MAE/RMSE/R² + PICP/MPIW/PINAW/CWC at 90/95/99
- `bnn_predictions.csv` — per-sample mean, std, and interval bounds
