#!/usr/bin/env bash
# Train Bi-LSTM across one or more feature sets.
# Usage: bash scripts/run_bilstm.sh                # all 5 feature sets
#        bash scripts/run_bilstm.sh top50 top100   # subset

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

FEATURE_SETS=("$@")
if [ "${#FEATURE_SETS[@]}" -eq 0 ]; then
    FEATURE_SETS=(top50 top100 top150 top200 top300)
fi

for fs in "${FEATURE_SETS[@]}"; do
    log="${LOG_DIR}/bilstm_${fs}.log"
    echo "[run]  Bi-LSTM @ ${fs} -> ${log}"
    python -u "${SRC_DIR}/03_train_eval.py" \
        --model "Bi-LSTM" --features "${fs}" 2>&1 | tee "${log}"
done
