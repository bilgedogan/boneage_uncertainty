# RSNA Pediatric Bone Age Estimation

Multi-input regression model for pediatric bone age estimation. Supports multiple backbones and reproducible seed-based experiments.

## Project Structure
```
├── config.py           # Paths and global constants
├── data_loader.py      # Data loading, splitting, DataLoaders
├── model.py            # Multi-input model (backbone + sex branch)
├── metrics.py          # Evaluation and CSV export
├── train.py            # Training loop entry point
├── run_training.sh     # Runs all 10 seeds for one backbone
├── tests/
│   └── test_smoke.py
├── .env                # DATA_DIR, OUTPUT_DIR
└── requirements.txt
├── data/
│   └── train/          # training images
│   └── val/            # validation images (for test)
│   └── train.csv
│   └── val.csv
```

## Setup

```bash
conda create -n uq python=3.11
conda activate uq
pip install -r requirements.txt
```

Configure `.env`:
```
DATA_DIR=/path/to/rsna/data
OUTPUT_DIR=/path/to/outputs
```

## Experiment Design

### Seeds
Seeds 0–9 control **both** network initialization (torch) and the **data split**. Each seed produces a distinct train/val/calibration partition so results capture both sources of randomness.

### Backbones
Three backbones are supported:

| Backbone | Input size | Feature dim |
|---|---|---|
| `efficientnet_b3` | 300×300 | 1536 |
| `vit_b16` | 224x224 | 768 |
| `convnext_tiny` | 300x300 | 768 |

All backbones use ImageNet pretrained weights. The sex branch (1 → 32) is concatenated with image features before the regression head (256 → 1).

### Data Split
Source training set → 50% train / 25% val / 25% calibration (stratified by sex).  
Source validation set → held-out test set (never seen during training).

## Running Experiments

### Run all 10 seeds for one backbone
```bash
bash run_training.sh --backbone efficientnet_b3
bash run_training.sh --backbone vit_b16
bash run_training.sh --backbone convnext_tiny
```

### Quick test (1% data, 2 epochs)
```bash
bash run_training.sh --backbone efficientnet_b3 --quick-test
```

### Run a single seed manually
```bash
python train.py --backbone vit_b16 --seed 3
python train.py --backbone convnext_tiny --seed 0 --quick-test
```

## Output Structure

```
OUTPUT_DIR/
└── <backbone>/
    └── seed<N>/
        ├── best_model.pth                    # Best checkpoint (lowest val MAE)
        ├── history.pkl                       # Per-epoch losses
        ├── loss_plot.png                     # Train vs val loss curve
        ├── val_predictions_<TS>.csv          # Val set: true vs predicted age
        ├── val_metrics_<TS>.csv              # Val set: MAE, RMSE, R², etc.
        ├── test_predictions_<TS>.csv         # Test set predictions
        └── test_metrics_<TS>.csv             # Test set metrics
```

### `history.pkl`
Python dict with per-epoch lists:
- `loss` — train Huber (Smooth L1) loss
- `val_loss` — validation MAE in normalized units
- `val_mae` — same as `val_loss` (alias)

### Metrics CSVs
Each row contains: `seed`, `MAE`, `RMSE`, `MSE`, `R2`, `pct_within_6mo`, `pct_within_12mo`, `n_val`, `timestamp`.  
All age values are in months (de-normalized).

### Loss functions
- **Training loss:** Smooth L1 (Huber) — robust to outliers during weight updates.
- **Validation/checkpoint metric:** MAE — interpretable in months, used for model selection.
