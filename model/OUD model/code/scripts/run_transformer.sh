#!/usr/bin/env bash
# Train Transformer encoder across one or more feature sets.
# Usage: bash scripts/run_transformer.sh                # all 5 feature sets
#        bash scripts/run_transformer.sh top50 top100   # subset

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

FEATURE_SETS=("$@")
if [ "${#FEATURE_SETS[@]}" -eq 0 ]; then
    FEATURE_SETS=(top50 top100 top150 top200 top300)
fi

for fs in "${FEATURE_SETS[@]}"; do
    log="${LOG_DIR}/transformer_${fs}.log"
    echo "[run]  Transformer @ ${fs} -> ${log}"
    python -u "${SRC_DIR}/03_train_eval.py" \
        --model "Transformer" --features "${fs}" 2>&1 | tee "${log}"
done
