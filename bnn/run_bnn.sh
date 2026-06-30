#!/bin/bash
# Runs BNN training and UQ evaluation for multiple seeds.
# efficientnet_b3, vit_b_16, convnextv2_tiny

BACKBONE="efficientnet_b3"
SEEDS=(0 1 2 3 4 5 6 7 8 9)
N_PASSES=60
PRIOR_SIGMA=1.0
GPU_ID="0"
COVERAGE=[0.90,0.95,0.99]

QUICK_TEST=0
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --quick-test) QUICK_TEST=1; shift ;;
        --backbone)   BACKBONE="$2"; shift 2 ;;
        --n-passes)   N_PASSES="$2"; shift 2 ;;
        --prior-sigma) PRIOR_SIGMA="$2"; shift 2 ;;
        --gpu)        GPU_ID="$2"; shift 2 ;;
        --coverage)   COVERAGE="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Set the GPU for this execution
export CUDA_VISIBLE_DEVICES=$GPU_ID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train.py"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=== GPU=$GPU_ID backbone=$BACKBONE seed=$SEED passes=$N_PASSES prior_sigma=$PRIOR_SIGMA coverage=$COVERAGE ==="

    PYTHON_CMD="python $TRAIN_SCRIPT --seed $SEED --backbone $BACKBONE --n-passes $N_PASSES --prior-sigma $PRIOR_SIGMA --coverage $COVERAGE"
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
echo "Aggregating results for backbone=$BACKBONE..."
python "$SCRIPT_DIR/aggregate_results.py" --backbone "$BACKBONE" --split test

echo ""
echo "All BNN runs completed for backbone=$BACKBONE."
