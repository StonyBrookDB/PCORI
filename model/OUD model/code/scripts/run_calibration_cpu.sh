#!/usr/bin/env bash
# CALIBRATION — CPU models only (no GPU): Logistic Regression, Decision Tree,
# Random Forest. Safe to run alongside a GPU job since these never touch the GPU.
#
# Runs the 3 threshold strategies (default 0.5 / f1_swept / prevalence_matched)
# and writes summary_calibration.json + fold_predictions.npz per experiment.
# Never overwrites the main summary.json / fold_metrics.csv.
#
# Usage:
#   bash scripts/run_calibration_cpu.sh                  # all 5 feature sets
#   bash scripts/run_calibration_cpu.sh top100 top300    # subset
#   bash scripts/run_calibration_cpu.sh --prevalence 0.01  # force fixed 1%

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
set +e

CPU_MODELS=("Logistic Regression" "Decision Tree" "Random Forest")

PREVALENCE=""
FEATURES=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --prevalence) shift; PREVALENCE="$1"; shift ;;
        *) FEATURES+=("$1"); shift ;;
    esac
done
[ "${#FEATURES[@]}" -eq 0 ] && FEATURES=(top50 top100 top150 top200 top300)
PREV_ARG=(); [ -n "${PREVALENCE}" ] && PREV_ARG=(--prevalence "${PREVALENCE}")

slug() { echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]\{1,\}/_/g; s/^_//; s/_$//'; }

echo "=== CALIBRATION (CPU models) — $(date '+%H:%M:%S') ==="
for fs in "${FEATURES[@]}"; do
    for model in "${CPU_MODELS[@]}"; do
        out="${PROJECT_DIR}/experiments/$(slug "${model}")_${fs}/summary_calibration.json"
        if [ -f "${out}" ]; then
            echo "[skip] ${model} @ ${fs} (summary_calibration.json exists)"
            continue
        fi
        log="${LOG_DIR}/calib_$(slug "${model}")_${fs}.log"
        echo "[run]  ${model} @ ${fs} -> ${log}"
        python -u "${SRC_DIR}/05_calibration_eval.py" \
            --model "${model}" --features "${fs}" "${PREV_ARG[@]}" > "${log}" 2>&1 \
            || echo "[FAIL] ${model} @ ${fs} (see ${log})"
    done
    python -u "${SRC_DIR}/05_calibration_eval.py" --summarize --features "${fs}"
done
echo "=== CPU calibration done — $(date '+%H:%M:%S') ==="
