#!/usr/bin/env bash
# Run every registered model across all 5 feature sets (top50/100/150/200/300).
#
# Add a new model:
#   1. Register the factory in src/model_registry.py
#   2. Copy scripts/run_<existing>.sh to scripts/run_<new>.sh (change the --model arg)
#   3. Add the script name to MODEL_SCRIPTS below
#
# Usage:
#   bash scripts/run_all.sh                                # everything
#   bash scripts/run_all.sh top50 top100                   # subset of feature sets

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Register all per-model scripts here. Order = run order.
MODEL_SCRIPTS=(
    # tabular baselines
    run_logistic_regression.sh
    run_decision_tree.sh
    run_random_forest.sh
    run_dnn.sh
    # Phase 1 — sequence models
    run_lstm.sh
    run_bilstm.sh
    run_attention.sh
    run_transformer.sh
    # Phase 2 (graph) and Phase 3 (hybrid) appended once implemented
)

# Feature sets passed through to each per-model script
FEATURE_SETS=("$@")
if [ "${#FEATURE_SETS[@]}" -eq 0 ]; then
    FEATURE_SETS=(top50 top100 top150 top200 top300)
fi

SCRIPTS_DIR="$(dirname "${BASH_SOURCE[0]}")"

for s in "${MODEL_SCRIPTS[@]}"; do
    echo
    echo "############################################################"
    echo "# ${s}  feature_sets=${FEATURE_SETS[*]}"
    echo "############################################################"
    bash "${SCRIPTS_DIR}/${s}" "${FEATURE_SETS[@]}"
done

# Final cross-model comparison tables (one CSV per feature set)
bash "${SCRIPTS_DIR}/summarize.sh" "${FEATURE_SETS[@]}"
