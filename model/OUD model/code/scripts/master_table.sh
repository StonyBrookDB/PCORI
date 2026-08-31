#!/usr/bin/env bash
# Build the master results table (all models × all feature sets × all metrics,
# main + calibration). Unfinished cells show as '-'. Read-only; safe anytime.
#
# Usage: bash scripts/master_table.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
python -u "${SRC_DIR}/build_master_table.py"
