"""
Backfill extended metrics (Sensitivity / Specificity / PPV / NPV / AUPRC +
balanced-test block) for already-finished experiments — WITHOUT retraining
and WITHOUT touching any existing file.

For each experiments/{exp}/ that has fold_predictions.npz (written by
src/05_calibration_eval.py), this script reproduces the metric set that the
updated src/03_train_eval.py.evaluate_fold() would have produced if those
experiments were re-run with the new code. The input is just the saved
per-fold (y_true, y_score) pairs.

Output: experiments/{exp}/summary_extended.json (new file only).

The original summary.json, summary_calibration.json, fold_metrics.csv etc.
are READ-ONLY here and are never modified.

Usage:
    python src/06_backfill_extended_metrics.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).parent.parent
EXP = ROOT / "experiments"
RANDOM_STATE = 42


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _clinical_metrics_at_05(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) else 0.0
    npv  = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0
    return {"sensitivity": float(sens), "specificity": float(spec),
            "ppv": float(ppv), "npv": float(npv), "f1": float(f1)}


def _balanced_subsample(y_true, y_score, fold: int):
    rng = np.random.default_rng(RANDOM_STATE + fold + 100)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if len(pos) == 0 or len(neg) == 0:
        return y_true, y_score
    n_keep_neg = min(len(neg), len(pos))
    neg_sample = rng.choice(neg, size=n_keep_neg, replace=False)
    keep = np.concatenate([pos, neg_sample])
    return y_true[keep], y_score[keep]


def _best_f1_threshold(y_true, y_score):
    p, r, t = precision_recall_curve(y_true, y_score)
    p, r = p[:-1], r[:-1]
    f1s = 2 * p * r / np.clip(p + r, 1e-12, None)
    if len(f1s) == 0:
        return 0.5
    return float(t[int(np.argmax(f1s))])


def metrics_for_fold(y_true: np.ndarray, y_score: np.ndarray, fold: int) -> dict:
    """Same signature/keys as src/03_train_eval.py.evaluate_fold() output,
    computed from saved predictions."""
    y_pred = (y_score >= 0.5).astype(np.int8)
    clin = _clinical_metrics_at_05(y_true, y_pred)

    auroc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else 0.0
    auprc = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else 0.0

    best_thr = _best_f1_threshold(y_true, y_score)
    y_best = (y_score >= best_thr).astype(np.int8)
    best = _clinical_metrics_at_05(y_true, y_best)

    yt_bal, ys_bal = _balanced_subsample(y_true, y_score, fold)
    yp_bal = (ys_bal >= 0.5).astype(np.int8)
    bal = _clinical_metrics_at_05(yt_bal, yp_bal)
    bal_auroc = float(roc_auc_score(yt_bal, ys_bal)) if len(np.unique(yt_bal)) == 2 else 0.0
    bal_auprc = float(average_precision_score(yt_bal, ys_bal)) if len(np.unique(yt_bal)) == 2 else 0.0

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": auroc,
        "sensitivity": clin["sensitivity"],
        "specificity": clin["specificity"],
        "ppv": clin["ppv"],
        "npv": clin["npv"],
        "auprc": auprc,
        "best_threshold": best_thr,
        "precision_at_best": best["ppv"],
        "recall_at_best": best["sensitivity"],
        "f1_best": best["f1"],
        "balanced_precision": bal["ppv"],
        "balanced_recall": bal["sensitivity"],
        "balanced_f1": bal["f1"],
        "balanced_specificity": bal["specificity"],
        "balanced_npv": bal["npv"],
        "balanced_auroc": bal_auroc,
        "balanced_auprc": bal_auprc,
    }


def backfill_one(exp_dir: Path):
    npz_path = exp_dir / "fold_predictions.npz"
    out_path = exp_dir / "summary_extended.json"
    if not npz_path.exists():
        return False, "no predictions"

    z = np.load(npz_path)
    folds = z["fold"]
    y_true_all = z["y_true"].astype(np.int8)
    y_score_all = z["y_score"].astype(np.float32)

    fold_results = []
    for k in np.unique(folds):
        sel = folds == k
        m = metrics_for_fold(y_true_all[sel], y_score_all[sel], fold=int(k) - 1)
        fold_results.append(m)

    summary = {"n_folds": len(fold_results),
               "source": "backfilled from fold_predictions.npz"}
    for metric in fold_results[0].keys():
        vals = np.array([r[metric] for r in fold_results])
        summary[f"{metric}_mean"] = round(float(vals.mean()), 4)
        summary[f"{metric}_std"] = round(float(vals.std()), 4)

    out_path.write_text(json.dumps(summary, indent=2))
    return True, f"AUROC={summary['auroc_mean']:.4f} bal_F1={summary['balanced_f1_mean']:.4f}"


def main():
    n_done = n_skip = 0
    for exp_dir in sorted(EXP.iterdir()):
        if not exp_dir.is_dir():
            continue
        ok, msg = backfill_one(exp_dir)
        if ok:
            print(f"  ✓ {exp_dir.name:<40} {msg}")
            n_done += 1
        else:
            print(f"  - {exp_dir.name:<40} {msg}")
            n_skip += 1
    print(f"\nBackfilled: {n_done}   skipped: {n_skip}")


if __name__ == "__main__":
    main()
