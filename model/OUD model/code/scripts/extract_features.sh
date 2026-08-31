#!/usr/bin/env bash
# Generate top50/top100/top150/top200/top300 feature lists from features.xlsx.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python -u "${SRC_DIR}/01_extract_features.py"
