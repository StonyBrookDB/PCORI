#!/usr/bin/env bash
# Generate comparison plots from completed experiments.
#
# Usage:
#   bash scripts/plot_results.sh                                    # all registered models
#   bash scripts/plot_results.sh --models "Random Forest" DNN       # subset
#   bash scripts/plot_results.sh --models "Logistic Regression" \
#       "Decision Tree" "Random Forest" DNN --suffix _baselines     # baselines only, custom suffix

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python -u "${SRC_DIR}/plot_results.py" "$@"

echo
echo "PNGs in results/:"
ls -lh "${PROJECT_DIR}/results/"*.png 2>/dev/null || true
