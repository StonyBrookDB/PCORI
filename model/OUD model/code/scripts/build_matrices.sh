#!/usr/bin/env bash
# Build patient × ICD-10 sparse binary matrices for one or all feature sets.
# Skips any matrix that already exists.
#
# Usage:
#   bash scripts/build_matrices.sh              # builds all: top50, top100, top150, top200
#   bash scripts/build_matrices.sh top50 top100 # builds the listed sets only

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

FEATURE_SETS=("$@")
if [ "${#FEATURE_SETS[@]}" -eq 0 ]; then
    FEATURE_SETS=(top50 top100 top150 top200 top300)
fi

for fs in "${FEATURE_SETS[@]}"; do
    out="${PROJECT_DIR}/data/matrix_${fs}.npz"
    log="${LOG_DIR}/build_${fs}.log"
    if [ -f "${out}" ]; then
        echo "[skip] ${fs}: matrix already exists at ${out}"
        continue
    fi
    echo "[run]  building matrix for ${fs} -> ${log}"
    python -u "${SRC_DIR}/02_build_matrix.py" --features "${fs}" 2>&1 | tee "${log}"
done

echo "Matrices present:"
ls -lh "${PROJECT_DIR}/data/"matrix_*.npz "${PROJECT_DIR}/data/labels.npy"
