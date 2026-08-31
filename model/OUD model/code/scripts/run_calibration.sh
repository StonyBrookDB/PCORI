#!/usr/bin/env bash
# Calibration / threshold-strategy comparison (separate experiment).
# Computes default(0.5) / f1_swept / prevalence_matched thresholds per model,
# writing summary_calibration.json — never touches the main summary.json.
#
# Usage:
#   bash scripts/run_calibration.sh                       # all models × all 5 feature sets
#   bash scripts/run_calibration.sh --models "Random Forest" DNN
#   bash scripts/run_calibration.sh --features top100 top300
#   bash scripts/run_calibration.sh --prevalence 0.01     # force a fixed 1% prevalence
#   bash scripts/run_calibration.sh --summarize           # just rebuild result tables

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
set +e   # keep going past individual failures

ALL_MODELS=("Logistic Regression" "Decision Tree" "Random Forest" "DNN"
            "LSTM" "Bi-LSTM" "Attention" "Transformer")
ALL_FEATURES=(top50 top100 top150 top200 top300)

MODELS=()
FEATURES=()
PREVALENCE=""
SUMMARIZE_ONLY=0

# ── simple arg parsing ──
while [ "$#" -gt 0 ]; do
    case "$1" in
        --models)     shift; while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
        --features)   shift; while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do FEATURES+=("$1"); shift; done ;;
        --prevalence) shift; PREVALENCE="$1"; shift ;;
        --summarize)  SUMMARIZE_ONLY=1; shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

[ "${#MODELS[@]}" -eq 0 ] && MODELS=("${ALL_MODELS[@]}")
[ "${#FEATURES[@]}" -eq 0 ] && FEATURES=("${ALL_FEATURES[@]}")

PREV_ARG=()
[ -n "${PREVALENCE}" ] && PREV_ARG=(--prevalence "${PREVALENCE}")

if [ "${SUMMARIZE_ONLY}" -eq 1 ]; then
    for fs in "${FEATURES[@]}"; do
        python -u "${SRC_DIR}/05_calibration_eval.py" --summarize --features "${fs}"
    done
    exit 0
fi

slug() { echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]\{1,\}/_/g; s/^_//; s/_$//'; }

for fs in "${FEATURES[@]}"; do
    for model in "${MODELS[@]}"; do
        out="${PROJECT_DIR}/experiments/$(slug "${model}")_${fs}/summary_calibration.json"
        if [ -f "${out}" ]; then
            echo "[skip] calib ${model} @ ${fs} (summary_calibration.json exists)"
            continue
        fi
        log="${LOG_DIR}/calib_$(slug "${model}")_${fs}.log"
        echo "[run]  calib ${model} @ ${fs} -> ${log}"
        python -u "${SRC_DIR}/05_calibration_eval.py" \
            --model "${model}" --features "${fs}" "${PREV_ARG[@]}" > "${log}" 2>&1 \
            || echo "[FAIL] calib ${model} @ ${fs} (see ${log})"
    done
    python -u "${SRC_DIR}/05_calibration_eval.py" --summarize --features "${fs}"
done
