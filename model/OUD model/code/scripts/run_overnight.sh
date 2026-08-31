#!/usr/bin/env bash
# Unified overnight runner.
#
# Pipeline:
#   1. Build data/sequences_top{N}.npy for all 5 feature sets (skips existing).
#   2. Train all Phase 1 sequence models (LSTM, Bi-LSTM, Attention, Transformer)
#      across all 5 feature sets — 20 (model, feature_set) combinations.
#   3. Generate one results/table2_top{N}.csv per feature set.
#
# Robustness features:
#   - Smart skip: any (model, feature_set) whose summary.json already exists
#     is skipped, so this script is safe to rerun after a crash.
#   - Per-run errors are caught — one failure does NOT stop the rest.
#   - Master log at logs/overnight.log captures everything; each individual
#     run also gets its own logs/{model_slug}_{feature_set}.log.
#
# Recommended invocation (so a disconnect can't kill it):
#   tmux new -s overnight 'bash scripts/run_overnight.sh'
#   # detach: Ctrl+B then D     reattach: tmux attach -t overnight

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Do NOT `set -e` for this script: we want to keep going past individual failures.
set +e

MODELS=("LSTM" "Bi-LSTM" "Attention" "Transformer")
FEATURE_SETS=(top50 top100 top150 top200 top300)
MASTER_LOG="${LOG_DIR}/overnight.log"

banner() {
    local line
    line="$(printf '=%.0s' {1..70})"
    echo "${line}"
    echo "  $1"
    echo "${line}"
}

# Compute the same slug that src/03_train_eval.py uses internally so we can
# check if a (model, feature_set) is already complete.
slug() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]\{1,\}/_/g; s/^_//; s/_$//'
}

T0=$(date +%s)
{
    banner "OVERNIGHT RUN START  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || echo "  (nvidia-smi unavailable)"
    echo
    echo "Plan:"
    echo "  Step 1: build_sequences for ${FEATURE_SETS[*]}"
    echo "  Step 2: ${#MODELS[@]} models × ${#FEATURE_SETS[@]} feature sets = $((${#MODELS[@]} * ${#FEATURE_SETS[@]})) experiments"
    echo "  Step 3: summarize"
    echo
} | tee "${MASTER_LOG}"

# ── Step 1 ─────────────────────────────────────────────────────────────────
banner "STEP 1 / 3 — build sequences" | tee -a "${MASTER_LOG}"
bash "${PROJECT_DIR}/scripts/build_sequences.sh" 2>&1 | tee -a "${MASTER_LOG}"
SEQ_RC=${PIPESTATUS[0]}
if [ "${SEQ_RC}" -ne 0 ]; then
    echo "[FATAL] sequence build returned ${SEQ_RC}; sequence models cannot run." | tee -a "${MASTER_LOG}"
    exit 1
fi

# ── Step 2 ─────────────────────────────────────────────────────────────────
banner "STEP 2 / 3 — train sequence models" | tee -a "${MASTER_LOG}"

declare -a COMPLETED
declare -a SKIPPED
declare -a FAILED

for model in "${MODELS[@]}"; do
    model_slug=$(slug "${model}")
    for fs in "${FEATURE_SETS[@]}"; do
        summary="${PROJECT_DIR}/experiments/${model_slug}_${fs}/summary.json"
        log="${LOG_DIR}/${model_slug}_${fs}.log"

        if [ -f "${summary}" ]; then
            echo "[skip] ${model} @ ${fs}  (summary.json already exists)" | tee -a "${MASTER_LOG}"
            SKIPPED+=("${model} @ ${fs}")
            continue
        fi

        START=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[run]  ${model} @ ${fs}  started ${START}  ->  ${log}" | tee -a "${MASTER_LOG}"
        python -u "${SRC_DIR}/03_train_eval.py" \
            --model "${model}" --features "${fs}" > "${log}" 2>&1
        RC=$?

        END=$(date '+%H:%M:%S')
        if [ "${RC}" -eq 0 ] && [ -f "${summary}" ]; then
            line=$(grep -E "AUROC=" "${log}" | tail -1)
            echo "[done] ${model} @ ${fs}  finished ${END}  ${line}" | tee -a "${MASTER_LOG}"
            COMPLETED+=("${model} @ ${fs}")
        else
            echo "[FAIL] ${model} @ ${fs}  rc=${RC}  see ${log}" | tee -a "${MASTER_LOG}"
            FAILED+=("${model} @ ${fs}")
        fi
    done
done

# ── Step 3 ─────────────────────────────────────────────────────────────────
banner "STEP 3 / 3 — summarize" | tee -a "${MASTER_LOG}"
bash "${PROJECT_DIR}/scripts/summarize.sh" "${FEATURE_SETS[@]}" 2>&1 | tee -a "${MASTER_LOG}"

# ── Final report ───────────────────────────────────────────────────────────
T1=$(date +%s)
ELAPSED_SEC=$((T1 - T0))
ELAPSED_FMT=$(printf '%dh %dm' $((ELAPSED_SEC / 3600)) $(((ELAPSED_SEC % 3600) / 60)))

{
    echo
    banner "OVERNIGHT RUN END  $(date '+%Y-%m-%d %H:%M:%S')  (elapsed ${ELAPSED_FMT})"
    echo "Completed: ${#COMPLETED[@]} / $((${#MODELS[@]} * ${#FEATURE_SETS[@]}))"
    for x in "${COMPLETED[@]}"; do echo "  ✓ ${x}"; done
    echo
    echo "Skipped (already done before this run): ${#SKIPPED[@]}"
    for x in "${SKIPPED[@]}"; do echo "  - ${x}"; done
    echo
    if [ "${#FAILED[@]}" -gt 0 ]; then
        echo "FAILED: ${#FAILED[@]}"
        for x in "${FAILED[@]}"; do echo "  ✗ ${x}"; done
        echo "(check the per-run logs in ${LOG_DIR}/ for tracebacks)"
    fi
    echo
    echo "Generated summary tables:"
    ls -la "${PROJECT_DIR}/results/"table2_*.csv 2>/dev/null || echo "  (none)"
} | tee -a "${MASTER_LOG}"

# Exit code: 0 if nothing failed, 1 otherwise (so cron / wrappers can detect)
[ "${#FAILED[@]}" -eq 0 ]
