#!/usr/bin/env bash
# Plot threshold-strategy comparison (F1-optimal vs prevalence-matched vs 0.5)
# from the calibration results. Read-only; safe anytime. Only models that have
# summary_calibration.json are drawn.
#
# Usage:
#   bash scripts/plot_calibration_compare.sh                  # all feature sets
#   bash scripts/plot_calibration_compare.sh --features top300

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
python -u "${SRC_DIR}/plot_calibration_compare.py" "$@"
echo
echo "calibration comparison PNGs:"
ls -lh "${PROJECT_DIR}/results/"calib_compare_*.png 2>/dev/null || true
