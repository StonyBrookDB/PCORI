#!/usr/bin/env bash
# Shared setup for all scripts. Sourced (not executed) from other scripts.

set -euo pipefail

CONDA_ENV="pcori_oud_2026"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${PROJECT_DIR}/src"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "${LOG_DIR}"

# Resolve conda; fall back to the user's miniconda install if not on PATH.
if ! command -v conda >/dev/null 2>&1; then
    export PATH="${HOME}/miniconda3/condabin:${PATH}"
fi

# Activate the conda env so subsequent `python` calls use it.
# Using `conda run` would also work but `activate` is friendlier for interactive logs.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"
