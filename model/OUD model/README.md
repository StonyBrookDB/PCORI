# OUD Model — Logistic Regression & Transformer

Code, trained model weights, and evaluation results for the Logistic Regression
and Transformer models in the OUD (opioid use disorder) prediction pipeline.
Raw patient data is intentionally **not** included in this folder.

## Contents

- `code/src/` — full model pipeline source (feature extraction, matrix/sequence
  building, training/eval, calibration, plotting). Model-specific logic for LR
  and Transformer lives in `03_train_eval.py`, `model_registry.py`,
  `seq_models.py`, and `08_calibration_split_eval.py`.
- `code/scripts/` — shell entry points for running each stage
  (`run_logistic_regression.sh`, `run_transformer.sh`, `run_exp_calib_july.sh`, etc.).
- `models/` — trained weights: `logistic_regression_*.pkl` and `transformer_*.pkl`,
  one per feature-set curation (`kelly`, `rick_top50`…`rick_top251`, `top50`…`top300`).
- `results/exp_calib_july/` — latest evaluation results (added 2026-07-27), from a
  proper train/calibration/test split that fixes a threshold-selection leak present
  in the earlier pipelines (see `docs/exp_calib_july.txt` for the full writeup).
  Includes per-curation summary CSVs, ROC/calibration figures, and confusion tables
  for both models.
- `docs/` — methodology notes: `exp_calib_july.txt` (latest split/calibration
  experiment), `model_logistic_regression.txt`, `model_transformer.txt`.

## Running

See `code/scripts/run_logistic_regression.sh`, `code/scripts/run_transformer.sh`,
and `code/scripts/run_exp_calib_july.sh`. These expect the feature matrices /
sequence tensors built by `code/src/01_extract_features.py`–`04_build_sequences.py`
from the original (private) patient-level data, which is not part of this upload.
