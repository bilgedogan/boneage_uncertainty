#!/bin/bash
# Runs split / kNN / RF conformal prediction for multiple seeds.
# efficientnet_b3, vit_b_16, convnextv2_tiny

BACKBONE="efficientnet_b3"
SEEDS=(0 1 2 3 4 5 6 7 8 9)
KNN_K=25
RF_TREES=200
RF_MIN_LEAF=5
COVERAGE=[0.90,0.95,0.99]

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --backbone)   BACKBONE="$2"; shift 2 ;;
        --knn-k)      KNN_K="$2"; shift 2 ;;
        --rf-trees)   RF_TREES="$2"; shift 2 ;;
        --rf-min-leaf) RF_MIN_LEAF="$2"; shift 2 ;;
        --coverage)   COVERAGE="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done


for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=== backbone=$BACKBONE seed=$SEED knn_k=$KNN_K rf_trees=$RF_TREES coverage=$COVERAGE ==="

    PYTHON_CMD="python cp/cp.py --seed $SEED --backbone $BACKBONE --knn-k $KNN_K --rf-trees $RF_TREES --rf-min-leaf $RF_MIN_LEAF --coverage $COVERAGE"

    eval "$PYTHON_CMD"

    if [ $? -ne 0 ]; then
        echo "ERROR: backbone=$BACKBONE seed=$SEED failed. Aborting."
        exit 1
    fi
done


echo ""
echo "All CP runs completed for backbone=$BACKBONE."
