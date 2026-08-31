#!/usr/bin/env bash
# Train Random Forest across one or more feature sets.
# Usage: bash scripts/run_random_forest.sh                # all 4 feature sets
#        bash scripts/run_random_forest.sh top50 top100   # subset

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

FEATURE_SETS=("$@")
if [ "${#FEATURE_SETS[@]}" -eq 0 ]; then
    FEATURE_SETS=(top50 top100 top150 top200 top300)
fi

for fs in "${FEATURE_SETS[@]}"; do
    log="${LOG_DIR}/rf_${fs}.log"
    echo "[run]  Random Forest @ ${fs} -> ${log}"
    python -u "${SRC_DIR}/03_train_eval.py" \
        --model "Random Forest" --features "${fs}" 2>&1 | tee "${log}"
done
