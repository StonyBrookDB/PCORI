#!/usr/bin/env bash
# Build patient-level ICD-10 sequence arrays for the sequence-based models.
# Skips a feature set if data/sequences_top{N}.npy already exists.
#
# Usage:
#   bash scripts/build_sequences.sh                     # all 5 sets
#   bash scripts/build_sequences.sh top50 top100        # subset

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

FEATURE_SETS=("$@")
if [ "${#FEATURE_SETS[@]}" -eq 0 ]; then
    FEATURE_SETS=(top50 top100 top150 top200 top300)
fi

# Pre-flight: which sets are still missing?
TO_BUILD=()
for fs in "${FEATURE_SETS[@]}"; do
    if [ -f "${PROJECT_DIR}/data/sequences_${fs}.npy" ]; then
        echo "[skip] ${fs}: data/sequences_${fs}.npy already exists"
    else
        TO_BUILD+=("${fs}")
    fi
done

if [ "${#TO_BUILD[@]}" -eq 0 ]; then
    echo "All requested sequence files already exist. Nothing to do."
    exit 0
fi

log="${LOG_DIR}/build_sequences.log"
echo "[run]  building sequences for: ${TO_BUILD[*]}  ->  ${log}"
python -u "${SRC_DIR}/04_build_sequences.py" --features "${TO_BUILD[@]}" 2>&1 | tee "${log}"

echo "Sequence files present:"
ls -lh "${PROJECT_DIR}/data/"sequences_*.npy 2>/dev/null || true
