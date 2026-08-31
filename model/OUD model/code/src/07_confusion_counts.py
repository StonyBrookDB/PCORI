"""
Compute TP / FN / FP / TN counts for the two paper models (Logistic Regression
and Transformer) across every feature set, on the BALANCED-test evaluation
(the operating point reported in Table 1 / Table 2).

Two paths depending on what artifacts each experiment has:

  (A) fold_predictions.npz exists  → recompute counts exactly:
        - regenerate the balanced subsample with the same seed
          (42 + fold + 100) as used during training/backfill
        - threshold = 0.5 on the saved y_score
        - sklearn.metrics.confusion_matrix gives the four counts

  (B) no fold_predictions.npz       → derive from saved per-fold metrics:
        - the fold's true positive count P_fold is deterministic from
          StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
          applied to labels.npy
        - balanced subset has P_fold positives + P_fold negatives
        - TP = round(balanced_recall_fold      * P_fold)
        - TN = round(balanced_specificity_fold * P_fold)
        - FN = P_fold - TP
        - FP = P_fold - TN

Both paths produce per-fold integer counts which are then summed across the
10 CV test folds (each patient appears in exactly one test fold, so the sum
covers the full dataset under the CV protocol).

Outputs:
  results/confusion_counts.csv      machine-readable, one row per
                                    (model, feature_set), with TP/FN/FP/TN
                                    (10-fold sum) and the derived sensitivity
                                    / specificity / PPV / NPV / F1 sanity check
  results/confusion_counts.tex      LaTeX three-line table for the paper

This script is read-only on every existing artifact. It writes only the two
new files above.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
EXP = ROOT / "experiments"
OUT = ROOT / "results"

MODELS = ["Logistic Regression", "Transformer"]
FEATURE_SETS = [
    # inclusive (already-done experiments)
    "top50", "top100", "top150", "top200", "top300",
    # Kelly limited
    "kelly",
    # Rick expanded scaling
    "rick_top50", "rick_top100", "rick_top150", "rick_top200", "rick_top251",
]
N_SPLITS = 10
RANDOM_STATE = 42


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# Per-fold positive counts (deterministic from labels + seed) ────────────────
_y = np.load(DATA / "labels.npy")
_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
FOLD_POS_COUNT = []   # P_fold for fold 0..9
FOLD_TEST_IDX = []
for _, test_idx in _skf.split(_y, _y):
    FOLD_TEST_IDX.append(test_idx)
    FOLD_POS_COUNT.append(int(_y[test_idx].sum()))


def _balanced_subsample(y_true: np.ndarray, y_score: np.ndarray, fold: int):
    """Reproduces src/03_train_eval.py._balanced_subsample exactly."""
    rng = np.random.default_rng(RANDOM_STATE + fold + 100)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if len(pos) == 0 or len(neg) == 0:
        return y_true, y_score
    n_keep_neg = min(len(neg), len(pos))
    neg_sample = rng.choice(neg, size=n_keep_neg, replace=False)
    keep = np.concatenate([pos, neg_sample])
    return y_true[keep], y_score[keep]


def counts_from_predictions(npz_path: Path) -> tuple[list[tuple[int, int, int, int]], str]:
    """Path (A): exact recomputation from saved per-fold predictions."""
    z = np.load(npz_path)
    folds = z["fold"]
    y_true = z["y_true"].astype(np.int8)
    y_score = z["y_score"].astype(np.float32)
    per_fold = []
    for k in sorted(np.unique(folds)):
        sel = folds == k
        yt, ys = _balanced_subsample(y_true[sel], y_score[sel], fold=int(k) - 1)
        yp = (ys >= 0.5).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        per_fold.append((int(tp), int(fn), int(fp), int(tn)))
    return per_fold, "from_predictions"


def counts_from_metrics(fold_csv: Path) -> tuple[list[tuple[int, int, int, int]], str]:
    """Path (B): derive from per-fold balanced_* metrics + StratifiedKFold positives."""
    per_fold = []
    with open(fold_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows or "balanced_recall" not in rows[0]:
        raise ValueError(f"{fold_csv} lacks balanced_* columns")
    for i, row in enumerate(rows):
        P = FOLD_POS_COUNT[i]
        sens = float(row["balanced_recall"])
        spec = float(row["balanced_specificity"])
        tp = int(round(sens * P))
        tn = int(round(spec * P))
        fn = P - tp
        fp = P - tn
        per_fold.append((tp, fn, fp, tn))
    return per_fold, "from_metrics"


def compute_one(model: str, fs: str):
    exp_dir = EXP / f"{slug(model)}_{fs}"
    npz = exp_dir / "fold_predictions.npz"
    fold_csv = exp_dir / "fold_metrics.csv"
    if npz.exists():
        return counts_from_predictions(npz)
    if fold_csv.exists():
        return counts_from_metrics(fold_csv)
    return None, "missing"


def safe_div(a, b):
    return a / b if b else 0.0


def main():
    rows = []
    for model in MODELS:
        for fs in FEATURE_SETS:
            result, source = compute_one(model, fs)
            if result is None:
                print(f"  - {model} @ {fs:<14} {source}")
                continue
            tp_sum = sum(t[0] for t in result)
            fn_sum = sum(t[1] for t in result)
            fp_sum = sum(t[2] for t in result)
            tn_sum = sum(t[3] for t in result)
            sens = safe_div(tp_sum, tp_sum + fn_sum)
            spec = safe_div(tn_sum, tn_sum + fp_sum)
            ppv  = safe_div(tp_sum, tp_sum + fp_sum)
            npv  = safe_div(tn_sum, tn_sum + fn_sum)
            f1   = safe_div(2 * ppv * sens, ppv + sens) if (ppv + sens) else 0.0
            rows.append({
                "model": model, "features": fs,
                "TP": tp_sum, "FN": fn_sum, "FP": fp_sum, "TN": tn_sum,
                "total": tp_sum + fn_sum + fp_sum + tn_sum,
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "ppv": round(ppv, 4),
                "npv": round(npv, 4),
                "f1": round(f1, 4),
                "source": source,
            })
            print(f"  ✓ {model:<22} {fs:<14}  "
                  f"TP={tp_sum:>7}  FN={fn_sum:>7}  FP={fp_sum:>7}  TN={tn_sum:>7}  "
                  f"({source})")

    # CSV
    csv_path = OUT / "confusion_counts.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {csv_path}")

    # LaTeX three-line table
    tex = ["\\begin{table*}[t]", "\\centering",
           "\\caption{Confusion-matrix counts for the two paper models on the "
           "class-balanced test evaluation (threshold 0.5), summed across the "
           "10 cross-validation test folds. TP = true positive, FN = false "
           "negative, FP = false positive, TN = true negative.}",
           "\\label{tab:confusion_counts}",
           "\\begin{tabular}{llrrrr}", "\\toprule",
           "Model & Feature set & TP & FN & FP & TN \\\\", "\\midrule"]
    last_model = None
    for r in rows:
        if last_model is not None and r["model"] != last_model:
            tex.append("\\midrule")
        last_model = r["model"]
        tex.append(f"{r['model'].replace('_', r' ')} & {r['features'].replace('_', r' ')} & "
                   f"{r['TP']:,} & {r['FN']:,} & {r['FP']:,} & {r['TN']:,} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    tex_path = OUT / "confusion_counts.tex"
    tex_path.write_text("\n".join(tex) + "\n")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
