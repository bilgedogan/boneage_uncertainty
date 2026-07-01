#!/bin/bash
# Train the Bayesian Neural Network (bayesian-torch) and run UQ inference
# across all seeds for one backbone.
#
# Backbones: efficientnet_b3, vit_b_16, convnextv2_tiny
#
# Usage:
#   bash bnn/run_bnn.sh --backbone efficientnet_b3 --n-passes 60 --gpu 0
#   bash bnn/run_bnn.sh --backbone vit_b_16 --quick-test --gpu 1

BACKBONE="efficientnet_b3"
SEEDS=(0 1 2 3 4 5 6 7 8 9)
N_PASSES=60
COVERAGE=[0.90,0.95,0.99]
GPU_ID="0"
QUICK_TEST=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --quick-test) QUICK_TEST=1; shift ;;
        --backbone)   BACKBONE="$2"; shift 2 ;;
        --n-passes)   N_PASSES="$2"; shift 2 ;;
        --coverage)   COVERAGE="$2"; shift 2 ;;
        --gpu)        GPU_ID="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Run from the project root so a relative .env (DATA_DIR/OUTPUT_DIR) is picked up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export CUDA_VISIBLE_DEVICES="$GPU_ID"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=== BNN backbone=$BACKBONE seed=$SEED passes=$N_PASSES coverage=$COVERAGE gpu=$GPU_ID ==="

    PYTHON_CMD="python bnn/bnn.py --seed $SEED --backbone $BACKBONE --n-passes $N_PASSES --coverage $COVERAGE"
    if [ "$QUICK_TEST" -eq 1 ]; then
        PYTHON_CMD="$PYTHON_CMD --quick-test"
    fi

    eval "$PYTHON_CMD"

    if [ $? -ne 0 ]; then
        echo "ERROR: backbone=$BACKBONE seed=$SEED failed. Aborting."
        exit 1
    fi
done

echo ""
echo "Done. Results in outputs/bnn/$BACKBONE/seed*/"
