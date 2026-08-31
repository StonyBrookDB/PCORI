"""
Calibration / threshold evaluation on a PROPER 3-way split — runs as a
SEPARATE experiment line ("exp_calib_july").

Both src/03_train_eval.py (f1_best) and src/05_calibration_eval.py
(f1_swept / prevalence_matched) select their operating threshold on the
SAME test fold used to report final metrics — see note/threshold_sweep.txt
and note/calibration.txt, which both flag this as an acknowledged leak.
This script fixes that with a proper train / calibration / test split:

  train        — fit the model (1:10 downsampled, same as 03/05).
  calibration  — independent of train and test, held at NATIVE prevalence
                 (~1.16%, NOT downsampled — see note/exp_calib_july.txt for
                 why). Used to fit isotonic/Platt probability calibrators
                 and to select operating thresholds. Never used to report
                 final metrics.
  test         — touched exactly once, only to compute final metrics.
                 Never used to fit the model, fit a calibrator, or select
                 a threshold.

For each (model, feature_set) this is a SINGLE split (no CV, no bootstrap
— deferred to a later iteration, see note/exp_calib_july.txt), scored under
3 calibration methods (raw / isotonic / platt) x 3 threshold strategies
(default / f1_swept / prevalence_matched) = 9 rows.

Outputs (NEVER touches experiments/{model}_{features}/ or results/calibration_*.csv):
  experiments/exp_calib_july/split_indices.npz          (shared across all runs)
  experiments/exp_calib_july/split_indices.meta.json
  experiments/exp_calib_july/{model_slug}_{fs}/summary_calib_split.json
  experiments/exp_calib_july/{model_slug}_{fs}/calib_split_metrics.csv
  experiments/exp_calib_july/{model_slug}_{fs}/predictions.npz
  results/exp_calib_july/{fs}.csv

Usage:
  python src/08_calibration_split_eval.py --model "Logistic Regression" --features top100
  python src/08_calibration_split_eval.py --all --features top100
  python src/08_calibration_split_eval.py --summarize --features top100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from model_registry import MODEL_REGISTRY, get_model, get_model_names, get_data_type

DATA_DIR = Path(__file__).parent.parent / "data"
EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments" / "exp_calib_july"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp_calib_july"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Match src/03_train_eval.py / src/05_calibration_eval.py so the trained
# model itself is produced identically; only the eval protocol differs.
RANDOM_STATE = 42
NEG_TO_POS_RATIO = 10

TRAIN_FRAC = 0.70
CALIB_FRAC = 0.15
TEST_FRAC = 0.15

CALIBRATION_METHODS = ["raw", "isotonic", "platt"]
STRATEGIES = ["default", "f1_swept", "prevalence_matched"]
N_BINS_RELIABILITY = 10

SPLIT_PATH = EXPERIMENTS_DIR / "split_indices.npz"
SPLIT_META_PATH = EXPERIMENTS_DIR / "split_indices.meta.json"


# ── helpers (duplicated from 03/05 so this script never disturbs them) ───────
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
        raise ValueError(f"Unsupported data_type for exp_calib_july: {data_type}")
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


# ── 3-way split (generated once, shared by every model / feature_set run) ────
def get_or_create_split(y: np.ndarray, seed: int = RANDOM_STATE):
    """Return (train_idx, calib_idx, test_idx), persisted to SPLIT_PATH so
    every model and every feature set is evaluated on the identical set of
    patients in each split (row order is shared across every matrix_*.npz /
    sequences_*.npy / labels.npy file, and the split only depends on y, so
    it never needs to be regenerated once it exists).
    """
    if SPLIT_PATH.exists():
        z = np.load(SPLIT_PATH)
        return z["train_idx"], z["calib_idx"], z["test_idx"]

    print(f"[split] No existing split at {SPLIT_PATH} — generating a new one.", flush=True)
    n = len(y)
    all_idx = np.arange(n)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(all_idx, y))

    calib_within_trainval = CALIB_FRAC / (TRAIN_FRAC + CALIB_FRAC)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=calib_within_trainval, random_state=seed + 1)
    train_pos, calib_pos = next(sss2.split(trainval_idx, y[trainval_idx]))
    train_idx = trainval_idx[train_pos]
    calib_idx = trainval_idx[calib_pos]

    # Atomic write: two concurrent invocations (e.g. two models launched in
    # separate terminals) must never observe a half-written split file.
    # NOTE: np.savez_compressed silently appends ".npz" to any filename that
    # doesn't already end in ".npz", so the temp name must end in ".npz"
    # itself or os.replace() below will look for a path numpy never wrote.
    tmp_path = SPLIT_PATH.parent / f"split_indices.tmp-{os.getpid()}.npz"
    np.savez_compressed(tmp_path, train_idx=train_idx, calib_idx=calib_idx, test_idx=test_idx)
    os.replace(tmp_path, SPLIT_PATH)

    meta = {
        "seed": seed,
        "train_frac": TRAIN_FRAC, "calib_frac": CALIB_FRAC, "test_frac": TEST_FRAC,
        "n_total": int(n),
        "n_train": int(len(train_idx)), "n_calib": int(len(calib_idx)), "n_test": int(len(test_idx)),
        "prevalence_total": float(y.mean()),
        "prevalence_train": float(y[train_idx].mean()),
        "prevalence_calib": float(y[calib_idx].mean()),
        "prevalence_test": float(y[test_idx].mean()),
    }
    SPLIT_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[split] Created: train={len(train_idx):,}  calib={len(calib_idx):,}  "
          f"test={len(test_idx):,}", flush=True)
    return train_idx, calib_idx, test_idx


# ── threshold strategies (same definitions as 05_calibration_eval.py) ────────
def threshold_for(strategy: str, y, y_score, prevalence: float) -> float:
    if strategy == "default":
        return 0.5
    if strategy == "f1_swept":
        precisions, recalls, thresholds = precision_recall_curve(y, y_score)
        p, r = precisions[:-1], recalls[:-1]
        f1s = 2 * p * r / np.clip(p + r, 1e-12, None)
        if len(f1s) == 0:
            return 0.5
        return float(thresholds[int(np.argmax(f1s))])
    if strategy == "prevalence_matched":
        prevalence = min(max(prevalence, 1e-6), 1.0)
        return float(np.quantile(y_score, 1.0 - prevalence))
    raise ValueError(f"Unknown strategy: {strategy}")


def metrics_at_threshold(y, y_score, thr) -> dict:
    y_pred = (y_score >= thr).astype(np.int8)
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "pred_pos_rate": float(y_pred.mean()),
    }


# ── probability calibration ───────────────────────────────────────────────────
def fit_isotonic(raw_calib_score: np.ndarray, y_calib: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip").fit(raw_calib_score, y_calib)


def fit_platt(raw_calib_score: np.ndarray, y_calib: np.ndarray) -> LogisticRegression:
    # Classic from-scratch Platt scaling: a 1-D logistic regression of the
    # raw score against the true label. Deliberately NOT sklearn's
    # CalibratedClassifierCV, which wraps its own internal CV around model
    # fitting — that would fight the explicit 3-way split built here.
    return LogisticRegression(solver="lbfgs").fit(raw_calib_score.reshape(-1, 1), y_calib)


def apply_calibrator(method: str, calibrator, raw_score: np.ndarray) -> np.ndarray:
    if method == "raw":
        return raw_score
    if method == "isotonic":
        return calibrator.predict(raw_score)
    if method == "platt":
        return calibrator.predict_proba(raw_score.reshape(-1, 1))[:, 1]
    raise ValueError(f"Unknown calibration method: {method}")


def reliability_table(y_true, y_prob, n_bins: int = N_BINS_RELIABILITY) -> list[dict]:
    """Quantile-binned reliability table. Uniform [0,1] bins are useless at
    ~1.16% prevalence (almost all mass would land in the lowest bin), so bin
    edges are quantiles of y_prob instead."""
    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return []
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        rows.append({
            "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
            "mean_predicted": float(y_prob[mask].mean()),
            "frac_positive": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return rows


# ── core ───────────────────────────────────────────────────────────────────────
def run_split_eval(model_name: str, feature_name: str, X, y,
                    train_idx, calib_idx, test_idx,
                    prevalence_override: float | None) -> Path:
    out_dir = experiment_dir(model_name, feature_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  EXP_CALIB_JULY  Model: {model_name}   Features: {feature_name}")
    print(f"  Output: {out_dir}")

    t0 = time.time()

    X_train_full, y_train_full = X[train_idx], y[train_idx]
    X_train, y_train = downsample_train(X_train_full, y_train_full, NEG_TO_POS_RATIO, seed=RANDOM_STATE)

    X_calib, y_calib = X[calib_idx], y[calib_idx]   # NEVER downsampled — native prevalence
    X_test, y_test = X[test_idx], y[test_idx]       # touched only once, below

    print(f"    train: {X_train.shape[0]:,} (pos={int(y_train.sum()):,}, "
          f"downsampled from {X_train_full.shape[0]:,})")
    print(f"    calib: {X_calib.shape[0]:,} (prevalence={y_calib.mean():.4f}, native)")
    print(f"    test : {X_test.shape[0]:,} (prevalence={y_test.mean():.4f}, native)")

    model = get_model(model_name, feature_name)
    model.fit(X_train, y_train)

    def score(Xp):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(Xp)[:, 1]
        return model.decision_function(Xp)

    raw_calib_score = score(X_calib)
    raw_test_score = score(X_test)

    # AUROC is rank-based; monotonic calibration transforms don't change it,
    # so it's computed once on the raw test scores, not per calibration method.
    auroc_test = float(roc_auc_score(y_test, raw_test_score))

    calibrators = {
        "raw": None,
        "isotonic": fit_isotonic(raw_calib_score, y_calib),
        "platt": fit_platt(raw_calib_score, y_calib),
    }

    prevalence = prevalence_override if prevalence_override is not None else float(y_calib.mean())

    brier_test: dict[str, float] = {}
    reliability_test: dict[str, list] = {}
    methods_out: dict[str, dict] = {}
    csv_rows = []
    scores_by_method: dict[str, np.ndarray] = {}

    for method in CALIBRATION_METHODS:
        cal_calib_score = apply_calibrator(method, calibrators[method], raw_calib_score)
        cal_test_score = apply_calibrator(method, calibrators[method], raw_test_score)
        scores_by_method[method] = cal_test_score

        brier_test[method] = float(brier_score_loss(y_test, cal_test_score))
        reliability_test[method] = reliability_table(y_test, cal_test_score)

        methods_out[method] = {}
        line = f"    [{method:<9}]"
        for strat in STRATEGIES:
            # Threshold selection happens ONLY on the calibration split.
            thr = threshold_for(strat, y_calib, cal_calib_score, prevalence)
            # Applied exactly once to the untouched test split.
            m = metrics_at_threshold(y_test, cal_test_score, thr)
            methods_out[method][strat] = m
            csv_rows.append({
                "calibration_method": method, "strategy": strat,
                **m, "auroc": auroc_test, "brier": brier_test[method],
            })
            line += f"  {strat}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
        print(line, flush=True)

    elapsed = time.time() - t0

    # ── save per-strategy CSV (no fold column — single split) ──
    csv_path = out_dir / "calib_split_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["calibration_method", "strategy", "threshold", "precision",
                           "recall", "f1", "pred_pos_rate", "auroc", "brier"])
        writer.writeheader()
        for r in csv_rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    # ── save summary JSON ──
    summary = {
        "model": model_name,
        "features": feature_name,
        "split": {
            "seed": RANDOM_STATE,
            "train_frac": TRAIN_FRAC, "calib_frac": CALIB_FRAC, "test_frac": TEST_FRAC,
            "n_train_raw": int(X_train_full.shape[0]), "n_train_downsampled": int(X_train.shape[0]),
            "n_calib": int(X_calib.shape[0]), "n_test": int(X_test.shape[0]),
            "prevalence_calib": float(y_calib.mean()), "prevalence_test": float(y_test.mean()),
        },
        "prevalence_mode": "fixed" if prevalence_override is not None else "calib_split_actual",
        "prevalence_value": prevalence_override if prevalence_override is not None else None,
        "elapsed_seconds": round(elapsed, 1),
        "auroc_test": auroc_test,
        "brier_test": brier_test,
        "reliability_curve_test": reliability_test,
        "methods": methods_out,
    }
    (out_dir / "summary_calib_split.json").write_text(json.dumps(summary, indent=2))

    # ── save predictions for zero-retrain re-analysis ──
    np.savez_compressed(
        out_dir / "predictions.npz",
        test_idx=test_idx.astype(np.int64),
        y_true=y_test.astype(np.int8),
        raw_score=scores_by_method["raw"].astype(np.float16),
        isotonic_score=scores_by_method["isotonic"].astype(np.float16),
        platt_score=scores_by_method["platt"].astype(np.float16),
    )

    print(f"  AUROC={auroc_test:.4f}  Brier(raw/isotonic/platt)="
          f"{brier_test['raw']:.4f}/{brier_test['isotonic']:.4f}/{brier_test['platt']:.4f}  "
          f"({elapsed:.0f}s)")
    return out_dir


# ── summarize ──────────────────────────────────────────────────────────────────
def summarize(feature_name: str):
    rows = []
    for entry in MODEL_REGISTRY:
        model_name = entry[0]
        path = experiment_dir(model_name, feature_name) / "summary_calib_split.json"
        if not path.exists():
            continue
        rows.append((model_name, json.loads(path.read_text())))

    if not rows:
        print(f"No exp_calib_july results found for {feature_name}.")
        return

    print(f"\n{'='*130}")
    print(f"  EXP_CALIB_JULY SUMMARY — feature set: {feature_name}")
    print(f"{'='*130}")
    header = (f"{'Model':<20}  {'AUROC':>8}  {'Method':<10}  {'Strategy':<20}  "
              f"{'Thr':>7}  {'Precision':>10}  {'Recall':>9}  {'F1':>9}  {'Brier':>8}")
    print(header)
    print("-" * 130)
    for model_name, s in rows:
        for i, method in enumerate(CALIBRATION_METHODS):
            for j, strat in enumerate(STRATEGIES):
                e = s["methods"][method][strat]
                au = f"{s['auroc_test']:.4f}" if (i == 0 and j == 0) else ""
                mname = model_name if (i == 0 and j == 0) else ""
                brier = f"{s['brier_test'][method]:.4f}" if j == 0 else ""
                print(f"{mname:<20}  {au:>8}  {method:<10}  {strat:<20}  "
                      f"{e['threshold']:>7.4f}  {e['precision']:>10.4f}  "
                      f"{e['recall']:>9.4f}  {e['f1']:>9.4f}  {brier:>8}")
        print("-" * 130)

    out_path = RESULTS_DIR / f"{feature_name}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "AUROC", "CalibrationMethod", "Strategy",
                    "Threshold", "Precision", "Recall", "F1", "PredPosRate", "Brier"])
        for model_name, s in rows:
            for method in CALIBRATION_METHODS:
                for strat in STRATEGIES:
                    e = s["methods"][method][strat]
                    w.writerow([
                        model_name, s["auroc_test"], method, strat,
                        e["threshold"], e["precision"], e["recall"], e["f1"],
                        e["pred_pos_rate"], s["brier_test"][method],
                    ])
    print(f"\nSaved exp_calib_july table to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="top100",
                        help="Feature set name (reads data/{name}_features.txt etc.)")
    parser.add_argument("--prevalence", type=float, default=None,
                        help="Fixed prevalence for prevalence_matched (e.g. 0.01). "
                             "Default: the calibration split's own prevalence (~0.0116).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help=f"Model name. Choices: {get_model_names()}")
    group.add_argument("--all", action="store_true")
    group.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if args.summarize:
        summarize(args.features)
        return

    y_full = np.load(DATA_DIR / "labels.npy")
    train_idx, calib_idx, test_idx = get_or_create_split(y_full)

    if args.all:
        cache: dict[str, tuple] = {}
        for model_name, _, data_type in MODEL_REGISTRY:
            if data_type not in cache:
                cache[data_type] = load_data(args.features, data_type)
            X, y = cache[data_type]
            run_split_eval(model_name, args.features, X, y, train_idx, calib_idx, test_idx, args.prevalence)
        summarize(args.features)
    else:
        data_type = get_data_type(args.model)
        X, y = load_data(args.features, data_type)
        run_split_eval(args.model, args.features, X, y, train_idx, calib_idx, test_idx, args.prevalence)


if __name__ == "__main__":
    main()
