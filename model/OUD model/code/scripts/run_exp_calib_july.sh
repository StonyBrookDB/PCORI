#!/usr/bin/env bash
# exp_calib_july — proper train/calibration/test split, isotonic + Platt
# probability calibration, no CV, no bootstrap (see note/exp_calib_july.txt).
# Writes experiments/exp_calib_july/{model}_{features}/summary_calib_split.json
# — never touches experiments/{model}_{features}/ (03/05's output) or
# results/calibration_*.csv.
#
# Default scope is the pilot: Logistic Regression + Transformer, on the
# three curation families used elsewhere in this repo (see
# plot_kelly_rick_full299_by_model.py) — kelly (48 codes), rick_top251
# (Rick's expanded 251-code list), top300 (the inclusive "Full" list).
# Use --models / --features to expand to the full registry once verified.
#
# Usage:
#   bash scripts/run_exp_calib_july.sh                              # pilot: LR + Transformer @ kelly/rick_top251/top300
#   bash scripts/run_exp_calib_july.sh --models "Random Forest" DNN
#   bash scripts/run_exp_calib_july.sh --features top100 top300
#   bash scripts/run_exp_calib_july.sh --models "${ALL_MODELS[@]}" --features "${ALL_FEATURES[@]}"
#   bash scripts/run_exp_calib_july.sh --prevalence 0.01             # force a fixed 1% prevalence
#   bash scripts/run_exp_calib_july.sh --summarize                   # just rebuild result tables

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
set +e   # keep going past individual failures

ALL_MODELS=("Logistic Regression" "Decision Tree" "Random Forest" "DNN"
            "LSTM" "Bi-LSTM" "Attention" "Transformer")
ALL_FEATURES=(top50 top100 top150 top200 top300
              rick_top50 rick_top100 rick_top150 rick_top200 rick_top251
              kelly)

DEFAULT_MODELS=("Logistic Regression" "Transformer")
DEFAULT_FEATURES=(kelly rick_top251 top300)

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

# Unflagged default is the PILOT scope, not the full grid (deliberate
# deviation from run_calibration.sh, which defaults to everything).
[ "${#MODELS[@]}" -eq 0 ] && MODELS=("${DEFAULT_MODELS[@]}")
[ "${#FEATURES[@]}" -eq 0 ] && FEATURES=("${DEFAULT_FEATURES[@]}")

PREV_ARG=()
[ -n "${PREVALENCE}" ] && PREV_ARG=(--prevalence "${PREVALENCE}")

if [ "${SUMMARIZE_ONLY}" -eq 1 ]; then
    for fs in "${FEATURES[@]}"; do
        python -u "${SRC_DIR}/08_calibration_split_eval.py" --summarize --features "${fs}"
    done
    exit 0
fi

slug() { echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]\{1,\}/_/g; s/^_//; s/_$//'; }

for fs in "${FEATURES[@]}"; do
    for model in "${MODELS[@]}"; do
        out="${PROJECT_DIR}/experiments/exp_calib_july/$(slug "${model}")_${fs}/summary_calib_split.json"
        if [ -f "${out}" ]; then
            echo "[skip] exp_calib_july ${model} @ ${fs} (summary_calib_split.json exists)"
            continue
        fi
        log="${LOG_DIR}/exp_calib_july_$(slug "${model}")_${fs}.log"
        echo "[run]  exp_calib_july ${model} @ ${fs} -> ${log}"
        python -u "${SRC_DIR}/08_calibration_split_eval.py" \
            --model "${model}" --features "${fs}" "${PREV_ARG[@]}" > "${log}" 2>&1 \
            || echo "[FAIL] exp_calib_july ${model} @ ${fs} (see ${log})"
    done
    python -u "${SRC_DIR}/08_calibration_split_eval.py" --summarize --features "${fs}"
done
