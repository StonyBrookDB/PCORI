#!/usr/bin/env bash
# Print and save Table-2-style comparison tables from completed experiments.
# Generates one CSV per feature set under results/.
#
# Usage:
#   bash scripts/summarize.sh                # all 4 feature sets
#   bash scripts/summarize.sh top100         # one feature set

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

FEATURE_SETS=("$@")
if [ "${#FEATURE_SETS[@]}" -eq 0 ]; then
    FEATURE_SETS=(top50 top100 top150 top200 top300)
fi

for fs in "${FEATURE_SETS[@]}"; do
    echo
    python -u "${SRC_DIR}/03_train_eval.py" --summarize --features "${fs}"
done
