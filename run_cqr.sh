#!/bin/bash
# Runs CQR training/evaluation for all 10 seeds and confidence levels.

BACKBONE="efficientnet_b3"
SEEDS=(0 1 2 3 4 5 6 7 8 9)
CONFIDENCES=(0.90 0.95 0.99)
ALPHAS=(0.10 0.05 0.01)
QUICK_TEST=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --quick-test) QUICK_TEST=1; shift ;;
        --backbone) BACKBONE="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

for IDX in "${!CONFIDENCES[@]}"; do
    CONFIDENCE="${CONFIDENCES[$IDX]}"
    ALPHA="${ALPHAS[$IDX]}"

    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "=== CQR backbone=$BACKBONE confidence=$CONFIDENCE alpha=$ALPHA seed=$SEED quick_test=$QUICK_TEST ==="

        CQR_BACKBONE="$BACKBONE" \
        CQR_ALPHA="$ALPHA" \
        CQR_SEED="$SEED" \
        CQR_QUICK_TEST="$QUICK_TEST" \
        python -m cqr.train_cqr

        if [ $? -ne 0 ]; then
            echo "ERROR: CQR backbone=$BACKBONE confidence=$CONFIDENCE seed=$SEED failed. Aborting."
            exit 1
        fi
    done

    echo ""
    echo "Aggregating CQR metrics for backbone=$BACKBONE confidence=$CONFIDENCE..."
    CQR_BACKBONE="$BACKBONE" \
    CQR_ALPHA="$ALPHA" \
    CQR_QUICK_TEST="$QUICK_TEST" \
    CQR_AGGREGATE=1 \
    python -m cqr.train_cqr
done

echo ""
echo "All CQR runs completed for backbone=$BACKBONE."
