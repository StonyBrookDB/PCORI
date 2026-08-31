#!/usr/bin/env bash
# CALIBRATION — GPU models only: DNN, LSTM, Bi-LSTM, Attention, Transformer.
#
# IMPORTANT: only run ONE GPU job at a time. Make sure no other GPU training
# (e.g. a main run_*.sh of a neural model, or another copy of this script) is
# active before launching this. Check with: nvidia-smi
#
# Runs the 3 threshold strategies (default 0.5 / f1_swept / prevalence_matched)
# and writes summary_calibration.json + fold_predictions.npz per experiment.
# Never overwrites the main summary.json / fold_metrics.csv.
#
# Usage:
#   bash scripts/run_calibration_gpu.sh                  # all 5 feature sets
#   bash scripts/run_calibration_gpu.sh top100 top300    # subset
#   bash scripts/run_calibration_gpu.sh --prevalence 0.01

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
set +e

GPU_MODELS=("DNN" "LSTM" "Bi-LSTM" "Attention" "Transformer")

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

# Guard: warn if another training process is already on the GPU.
if ps aux | grep -E "03_train_eval|05_calibration" | grep -v grep | grep -vq "$$"; then
    echo "WARNING: another train/calibration python process seems to be running."
    echo "Running two GPU jobs at once can OOM the card. Current GPU usage:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
    echo "Continuing in 10s — Ctrl+C to abort..."
    sleep 10
fi

echo "=== CALIBRATION (GPU models) — $(date '+%H:%M:%S') ==="
for fs in "${FEATURES[@]}"; do
    for model in "${GPU_MODELS[@]}"; do
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
echo "=== GPU calibration done — $(date '+%H:%M:%S') ==="
