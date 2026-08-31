#!/usr/bin/env bash
# Train Attention (multi-head self-attention pooling) across one or more feature sets.
# Usage: bash scripts/run_attention.sh                # all 5 feature sets
#        bash scripts/run_attention.sh top50 top100   # subset

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

FEATURE_SETS=("$@")
if [ "${#FEATURE_SETS[@]}" -eq 0 ]; then
    FEATURE_SETS=(top50 top100 top150 top200 top300)
fi

for fs in "${FEATURE_SETS[@]}"; do
    log="${LOG_DIR}/attention_${fs}.log"
    echo "[run]  Attention @ ${fs} -> ${log}"
    python -u "${SRC_DIR}/03_train_eval.py" \
        --model "Attention" --features "${fs}" 2>&1 | tee "${log}"
done
