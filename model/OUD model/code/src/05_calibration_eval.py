"""
Calibration / threshold-strategy comparison — runs as a SEPARATE experiment.

Doctors require the operating threshold to be calibrated to disease prevalence.
This script re-runs the same 10-fold CV protocol (identical splits, identical
1:10 training downsampling, identical seeds) as src/03_train_eval.py, but for
each fold it scores the test set ONCE and then evaluates three threshold
strategies:

  1. default            — threshold = 0.5 (same as the main results)
  2. f1_swept           — F1-optimal threshold from the PR curve (our method)
  3. prevalence_matched — threshold set so the predicted positive rate equals
                          the test fold's true prevalence (~1.16%); i.e. flag
                          the top-(prevalence) highest-risk patients. This is
                          the clinician-requested "calibrate to prevalence"
                          operating point.

AUROC is threshold-independent and is reported once per fold.

Outputs (NEVER overwrites the main results):
  experiments/{model_slug}_{fs}/summary_calibration.json
  experiments/{model_slug}_{fs}/calibration_fold_metrics.csv
  experiments/{model_slug}_{fs}/fold_predictions.npz   (test idx / y_true / y_score
                                                        per fold, so any future
                                                        threshold can be recomputed
                                                        with zero retraining)

Usage:
  python src/05_calibration_eval.py --model "Random Forest" --features top100
  python src/05_calibration_eval.py --all --features top100
  python src/05_calibration_eval.py --summarize --features top100
  python src/05_calibration_eval.py --model DNN --features top300 --prevalence 0.01
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from scipy.stats import ttest_rel
from sklearn.metrics import (
    f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from model_registry import MODEL_REGISTRY, get_model, get_model_names, get_data_type

DATA_DIR = Path(__file__).parent.parent / "data"
EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
RESULTS_DIR = Path(__file__).parent.parent / "results"
EXPERIMENTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Match src/03_train_eval.py exactly so folds/downsampling are identical.
N_SPLITS = 10
RANDOM_STATE = 42
NEG_TO_POS_RATIO = 10

STRATEGIES = ["default", "f1_swept", "prevalence_matched"]


# ── helpers (kept self-contained so this script never disturbs 03_train_eval) ─
def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def experiment_dir(model_name: str, feature_name: str) -> Path:
    return EXPERIMENTS_DIR / f"{slug(model_name)}_{feature_name}"


def load_data(feature_name: str, data_type: str):
    if data_type == "matrix":
        X = load_npz(DATA_DIR / f"matrix_{feature_name}.npz")
    elif data_type == "sequence":
        X = np.load(DATA_DIR / f"sequences_{feature_name}.npy")
    else:
        raise ValueError(f"Unsupported data_type for calibration: {data_type}")
    y = np.load(DATA_DIR / "labels.npy")
    return X, y


def downsample_train(X_train, y_train, neg_to_pos_ratio: int, seed: int):
    """Identical to src/03_train_eval.py.downsample_train()."""
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    n_keep_neg = min(len(neg_idx), neg_to_pos_ratio * len(pos_idx))
    sampled_neg = rng.choice(neg_idx, size=n_keep_neg, replace=False)
    keep = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(keep)
    return X_train[keep], y_train[keep]


# ── threshold strategies ─────────────────────────────────────────────────────
def threshold_for(strategy: str, y_test, y_score, prevalence: float) -> float:
    if strategy == "default":
        return 0.5
    if strategy == "f1_swept":
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
        p, r = precisions[:-1], recalls[:-1]
        f1s = 2 * p * r / np.clip(p + r, 1e-12, None)
        if len(f1s) == 0:
            return 0.5
        return float(thresholds[int(np.argmax(f1s))])
    if strategy == "prevalence_matched":
        # Flag the top `prevalence` fraction of patients by score.
        # quantile at (1 - prevalence) makes predicted positive rate ≈ prevalence.
        prevalence = min(max(prevalence, 1e-6), 1.0)
        return float(np.quantile(y_score, 1.0 - prevalence))
    raise ValueError(f"Unknown strategy: {strategy}")


def metrics_at_threshold(y_test, y_score, thr) -> dict:
    y_pred = (y_score >= thr).astype(np.int8)
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "pred_pos_rate": float(y_pred.mean()),
    }


# ── core ─────────────────────────────────────────────────────────────────────
def run_calibration(model_name: str, feature_name: str, X, y,
                    prevalence_override: float | None) -> Path:
    out_dir = experiment_dir(model_name, feature_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  CALIBRATION  Model: {model_name}   Features: {feature_name}")
    print(f"  prevalence: {'test-fold actual' if prevalence_override is None else prevalence_override}")
    print(f"  Output: {out_dir}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # per-fold rows: list of dicts (one per fold per strategy) + auroc per fold
    fold_rows = []
    auroc_per_fold = []
    # prediction store for zero-retrain future threshold analysis
    pred_idx, pred_fold, pred_true, pred_score = [], [], [], []

    t0 = time.time()
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_full, y_train_full = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        X_train, y_train = downsample_train(
            X_train_full, y_train_full, NEG_TO_POS_RATIO, seed=RANDOM_STATE + fold
        )

        model = get_model(model_name, feature_name)
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        auroc = float(roc_auc_score(y_test, y_score))
        auroc_per_fold.append(auroc)
        prevalence = prevalence_override if prevalence_override is not None else float(y_test.mean())

        line = f"    fold {fold+1:2d}: AUROC={auroc:.4f}  prev={prevalence:.4f}"
        for strat in STRATEGIES:
            thr = threshold_for(strat, y_test, y_score, prevalence)
            m = metrics_at_threshold(y_test, y_score, thr)
            m.update({"fold": fold + 1, "strategy": strat, "auroc": auroc})
            fold_rows.append(m)
            line += f"  | {strat}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
        print(line, flush=True)

        # store predictions (compact dtypes)
        pred_idx.append(test_idx.astype(np.int64))
        pred_fold.append(np.full(len(test_idx), fold + 1, dtype=np.int8))
        pred_true.append(y_test.astype(np.int8))
        pred_score.append(y_score.astype(np.float16))

    elapsed = time.time() - t0

    # ── save per-fold CSV ──
    fold_csv = out_dir / "calibration_fold_metrics.csv"
    with open(fold_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["fold", "strategy", "threshold", "precision",
                           "recall", "f1", "pred_pos_rate", "auroc"])
        writer.writeheader()
        for r in fold_rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    # ── aggregate to summary_calibration.json ──
    summary = {
        "model": model_name,
        "features": feature_name,
        "n_splits": N_SPLITS,
        "prevalence_mode": ("fixed" if prevalence_override is not None else "test_fold_actual"),
        "prevalence_value": (prevalence_override if prevalence_override is not None else None),
        "elapsed_seconds": round(elapsed, 1),
        "auroc_mean": round(float(np.mean(auroc_per_fold)), 4),
        "auroc_std": round(float(np.std(auroc_per_fold)), 4),
        "strategies": {},
    }
    for strat in STRATEGIES:
        sub = [r for r in fold_rows if r["strategy"] == strat]
        entry = {}
        for metric in ["threshold", "precision", "recall", "f1", "pred_pos_rate"]:
            vals = np.array([r[metric] for r in sub])
            entry[f"{metric}_mean"] = round(float(vals.mean()), 4)
            entry[f"{metric}_std"] = round(float(vals.std()), 4)
        summary["strategies"][strat] = entry

    (out_dir / "summary_calibration.json").write_text(json.dumps(summary, indent=2))

    # ── save predictions for zero-retrain re-analysis ──
    np.savez_compressed(
        out_dir / "fold_predictions.npz",
        test_idx=np.concatenate(pred_idx),
        fold=np.concatenate(pred_fold),
        y_true=np.concatenate(pred_true),
        y_score=np.concatenate(pred_score),
    )

    # console recap
    for strat in STRATEGIES:
        e = summary["strategies"][strat]
        print(f"  → {strat:<20} thr={e['threshold_mean']:.4f}  "
              f"P={e['precision_mean']:.4f}  R={e['recall_mean']:.4f}  "
              f"F1={e['f1_mean']:.4f}  predPos={e['pred_pos_rate_mean']:.4f}")
    print(f"  AUROC={summary['auroc_mean']:.4f}±{summary['auroc_std']:.4f}  ({elapsed:.0f}s)")
    return out_dir


# ── summarize ────────────────────────────────────────────────────────────────
def summarize(feature_name: str):
    rows = []
    for entry in MODEL_REGISTRY:
        model_name = entry[0]
        path = experiment_dir(model_name, feature_name) / "summary_calibration.json"
        if not path.exists():
            continue
        rows.append((model_name, json.loads(path.read_text())))

    if not rows:
        print(f"No calibration results found for {feature_name}.")
        return

    print(f"\n{'='*120}")
    print(f"  CALIBRATION SUMMARY — feature set: {feature_name}")
    print(f"{'='*120}")
    header = (f"{'Model':<20}  {'AUROC':>8}  {'Strategy':<20}  "
             f"{'Thr':>7}  {'Precision':>10}  {'Recall':>9}  {'F1':>9}  {'PredPos':>8}")
    print(header)
    print("-" * 120)
    for model_name, s in rows:
        for i, strat in enumerate(STRATEGIES):
            e = s["strategies"][strat]
            au = f"{s['auroc_mean']:.4f}" if i == 0 else ""
            mname = model_name if i == 0 else ""
            print(f"{mname:<20}  {au:>8}  {strat:<20}  "
                  f"{e['threshold_mean']:>7.4f}  "
                  f"{e['precision_mean']:>10.4f}  {e['recall_mean']:>9.4f}  "
                  f"{e['f1_mean']:>9.4f}  {e['pred_pos_rate_mean']:>8.4f}")
        print("-" * 120)

    # save wide CSV
    out_path = RESULTS_DIR / f"calibration_{feature_name}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "AUROC_mean", "AUROC_std", "Strategy",
                    "Threshold_mean", "Precision_mean", "Precision_std",
                    "Recall_mean", "Recall_std", "F1_mean", "F1_std",
                    "PredPosRate_mean"])
        for model_name, s in rows:
            for strat in STRATEGIES:
                e = s["strategies"][strat]
                w.writerow([
                    model_name, s["auroc_mean"], s["auroc_std"], strat,
                    e["threshold_mean"], e["precision_mean"], e["precision_std"],
                    e["recall_mean"], e["recall_std"], e["f1_mean"], e["f1_std"],
                    e["pred_pos_rate_mean"],
                ])
    print(f"\nSaved calibration table to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="top100",
                        help="Feature set name (reads data/{name}_features.txt etc.)")
    parser.add_argument("--prevalence", type=float, default=None,
                        help="Fixed prevalence for prevalence_matched (e.g. 0.01). "
                             "Default: each test fold's own prevalence (~0.0116).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help=f"Model name. Choices: {get_model_names()}")
    group.add_argument("--all", action="store_true")
    group.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if args.summarize:
        summarize(args.features)
        return

    if args.all:
        cache: dict[str, tuple] = {}
        for model_name, _, data_type in MODEL_REGISTRY:
            if data_type not in cache:
                cache[data_type] = load_data(args.features, data_type)
            X, y = cache[data_type]
            run_calibration(model_name, args.features, X, y, args.prevalence)
        summarize(args.features)
    else:
        data_type = get_data_type(args.model)
        X, y = load_data(args.features, data_type)
        run_calibration(args.model, args.features, X, y, args.prevalence)


if __name__ == "__main__":
    main()
