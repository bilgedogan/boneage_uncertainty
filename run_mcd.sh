#!/bin/bash
# Runs MC Dropout inference for multiple seeds.
# efficientnet_b3, vit_b_16, convnextv2_tiny

BACKBONE="efficientnet_b3"
SEEDS=(0 1 2 3 4 5 6 7 8 9)
N_PASSES=60
CONFORMALIZED=0
COVERAGE=[0.90,0.95,0.99]

QUICK_TEST=0
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --quick-test) QUICK_TEST=1; shift ;;
        --backbone)   BACKBONE="$2"; shift 2 ;;
        --n-passes)   N_PASSES="$2"; shift 2 ;;
        --conformalized) CONFORMALIZED="$2"; shift 2 ;;
        --coverage)   COVERAGE="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done


for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=== backbone=$BACKBONE seed=$SEED passes=$N_PASSES conformalized=$CONFORMALIZED coverage=$COVERAGE ==="

    PYTHON_CMD="python mcd/mcd.py --seed $SEED --backbone $BACKBONE --n-passes $N_PASSES --conformalized $CONFORMALIZED  --coverage $COVERAGE"
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
echo "All MCD runs completed for backbone=$BACKBONE."
