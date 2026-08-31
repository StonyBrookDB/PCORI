#!/usr/bin/env bash
# Train Decision Tree across one or more feature sets.
# Usage: bash scripts/run_decision_tree.sh                # all 4 feature sets
#        bash scripts/run_decision_tree.sh top50 top100   # subset

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

FEATURE_SETS=("$@")
if [ "${#FEATURE_SETS[@]}" -eq 0 ]; then
    FEATURE_SETS=(top50 top100 top150 top200 top300)
fi

for fs in "${FEATURE_SETS[@]}"; do
    log="${LOG_DIR}/dt_${fs}.log"
    echo "[run]  Decision Tree @ ${fs} -> ${log}"
    python -u "${SRC_DIR}/03_train_eval.py" \
        --model "Decision Tree" --features "${fs}" 2>&1 | tee "${log}"
done
