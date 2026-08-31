"""
Train and evaluate one model for OUD prediction.

Results are written to:
  experiments/{model_slug}_{feature_name}/
    fold_metrics.csv   — per-fold Precision/Recall/F1/AUROC
    summary.json       — mean ± std aggregated metrics

Run a single model:
  python src/03_train_eval.py --model "Random Forest" --features top100

Run all models in the registry:
  python src/03_train_eval.py --all --features top100

Compare all completed experiments and print Table-2-style summary:
  python src/03_train_eval.py --summarize --features top100
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz
from scipy.stats import ttest_rel
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))
from model_registry import MODEL_REGISTRY, get_model, get_model_names, get_data_type

DATA_DIR = Path(__file__).parent.parent / "data"
EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
MODELS_DIR = Path(__file__).parent.parent / "models"
EXPERIMENTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

N_SPLITS = 10
RANDOM_STATE = 42

# Training-set negative downsampling. Test set keeps original 1.16% prevalence.
# See note/_general.txt for rationale.
NEG_TO_POS_RATIO = 10


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def experiment_dir(model_name: str, feature_name: str) -> Path:
    return EXPERIMENTS_DIR / f"{slug(model_name)}_{feature_name}"


def load_data(feature_name: str, data_type: str):
    """Load X for a given data_type, plus the shared label vector y.

    - matrix:    sparse CSR matrix (n_patients, n_features)
    - sequence:  int16 array       (n_patients, max_len)
    - graph:     (placeholder for Phase 2)
    """
    if data_type == "matrix":
        X = load_npz(DATA_DIR / f"matrix_{feature_name}.npz")
    elif data_type == "sequence":
        X = np.load(DATA_DIR / f"sequences_{feature_name}.npy")
    elif data_type == "graph":
        raise NotImplementedError("graph data_type is reserved for Phase 2")
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    y = np.load(DATA_DIR / "labels.npy")
    return X, y


def downsample_train(X_train, y_train, neg_to_pos_ratio: int, seed: int):
    """Keep all positives + (ratio × n_pos) randomly sampled negatives.

    Applied to the TRAINING fold only; the test fold keeps its original
    1.16% prevalence so reported metrics reflect the true distribution.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    n_keep_neg = min(len(neg_idx), neg_to_pos_ratio * len(pos_idx))
    sampled_neg = rng.choice(neg_idx, size=n_keep_neg, replace=False)
    keep = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(keep)
    return X_train[keep], y_train[keep]


def _best_f1_from_pr_curve(y_test, y_score) -> tuple[float, float, float, float]:
    """Sweep the PR curve and return (best_threshold, P, R, F1) at the
    F1-maximizing point. Threshold sweep is run on this fold's TEST set —
    standard practice when reporting "best achievable F1" in imbalanced
    classification papers. Note: this is an optimistic estimate; for
    deployment, threshold should be tuned on a held-out validation split.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
    # precisions/recalls have one extra entry (P=1, R=0 sentinel) vs thresholds
    p = precisions[:-1]
    r = recalls[:-1]
    denom = np.clip(p + r, 1e-12, None)
    f1s = 2 * p * r / denom
    if len(f1s) == 0:
        return 0.5, 0.0, 0.0, 0.0
    best = int(np.argmax(f1s))
    return float(thresholds[best]), float(p[best]), float(r[best]), float(f1s[best])


def _clinical_metrics_at_05(y_true, y_pred):
    """Sensitivity, Specificity, PPV, NPV, F1 at the supplied y_pred."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0   # = recall
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) else 0.0   # = precision
    npv  = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0
    return {"sensitivity": float(sens), "specificity": float(spec),
            "ppv": float(ppv), "npv": float(npv), "f1": float(f1)}


def _balanced_subsample(y_test, y_score, fold: int):
    """All positives + same-count random negatives from this fold's test set.
    Deterministic seed (distinct from train downsample so the subsets don't
    correlate). Returns (y_true_bal, y_score_bal)."""
    rng = np.random.default_rng(RANDOM_STATE + fold + 100)
    pos = np.flatnonzero(y_test == 1)
    neg = np.flatnonzero(y_test == 0)
    if len(pos) == 0 or len(neg) == 0:
        return y_test, y_score
    n_keep_neg = min(len(neg), len(pos))
    neg_sample = rng.choice(neg, size=n_keep_neg, replace=False)
    keep = np.concatenate([pos, neg_sample])
    return y_test[keep], y_score[keep]


def evaluate_fold(model, X_train, y_train, X_test, y_test, fold: int = 0) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        # LinearSVC: use decision_function
        y_score = model.decision_function(X_test)

    best_thr, best_p, best_r, best_f1 = _best_f1_from_pr_curve(y_test, y_score)

    # confusion-matrix-derived metrics at thr=0.5 (for completeness)
    clin = _clinical_metrics_at_05(y_test, y_pred)

    # balanced-test evaluation (paper-comparable F1/P/R)
    y_test_bal, y_score_bal = _balanced_subsample(y_test, y_score, fold)
    y_pred_bal = (y_score_bal >= 0.5).astype(np.int8)
    clin_bal = _clinical_metrics_at_05(y_test_bal, y_pred_bal)

    return {
        # at default threshold 0.5 (real-prevalence test set)
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_test, y_score)),
        # new clinical metrics at thr=0.5
        "sensitivity": clin["sensitivity"],
        "specificity": clin["specificity"],
        "ppv": clin["ppv"],
        "npv": clin["npv"],
        # threshold-independent
        "auprc": float(average_precision_score(y_test, y_score)),
        # F1-optimal threshold (this fold's test set)
        "best_threshold": best_thr,
        "precision_at_best": best_p,
        "recall_at_best": best_r,
        "f1_best": best_f1,
        # balanced-test (paper-comparable: F1 in the 0.7-0.8 range)
        "balanced_precision": clin_bal["ppv"],
        "balanced_recall": clin_bal["sensitivity"],
        "balanced_f1": clin_bal["f1"],
        "balanced_specificity": clin_bal["specificity"],
        "balanced_npv": clin_bal["npv"],
        "balanced_auroc": float(roc_auc_score(y_test_bal, y_score_bal))
        if len(np.unique(y_test_bal)) == 2 else 0.0,
        "balanced_auprc": float(average_precision_score(y_test_bal, y_score_bal))
        if len(np.unique(y_test_bal)) == 2 else 0.0,
    }


def run_model(model_name: str, feature_name: str, X, y) -> Path:
    out_dir = experiment_dir(model_name, feature_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model   : {model_name}")
    print(f"  Features: {feature_name}  ({X.shape[1]} features, {X.shape[0]:,} patients)")
    print(f"  Output  : {out_dir}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    t0 = time.time()
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_full, y_train_full = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        X_train, y_train = downsample_train(
            X_train_full, y_train_full,
            neg_to_pos_ratio=NEG_TO_POS_RATIO,
            seed=RANDOM_STATE + fold,
        )
        print(f"    fold {fold+1:2d}: train={X_train.shape[0]:,} "
              f"(pos={int(y_train.sum()):,}/{X_train.shape[0]:,}={y_train.mean():.4f}) "
              f"test={X_test.shape[0]:,} (pos_rate={y_test.mean():.4f})",
              flush=True)

        metrics = evaluate_fold(get_model(model_name, feature_name),
                                X_train, y_train, X_test, y_test, fold=fold)
        fold_results.append(metrics)
        print(f"    fold {fold+1:2d}: "
              f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
              f"F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}  "
              f"|  bal_F1={metrics['balanced_f1']:.4f}  "
              f"AUPRC={metrics['auprc']:.4f}",
              flush=True)

    elapsed = time.time() - t0

    # All metric keys emitted by evaluate_fold — pulled from the first fold so
    # the file stays in sync if the dict grows again.
    METRIC_KEYS = list(fold_results[0].keys())

    # Save per-fold CSV
    fold_csv = out_dir / "fold_metrics.csv"
    fieldnames = ["fold"] + METRIC_KEYS
    with open(fold_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(fold_results):
            writer.writerow({"fold": i + 1, **r})

    # Save summary JSON
    summary = {
        "model": model_name,
        "features": feature_name,
        "n_splits": N_SPLITS,
        "elapsed_seconds": round(elapsed, 1),
    }
    for metric in METRIC_KEYS:
        vals = [r[metric] for r in fold_results]
        summary[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
        summary[f"{metric}_std"] = round(float(np.std(vals)), 4)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    # test_metrics.json is identical content; both names kept for clarity
    # (every CV fold's metrics are computed on the held-out test fold).
    (out_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"  → F1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f}  "
          f"AUROC={summary['auroc_mean']:.4f}±{summary['auroc_std']:.4f}  "
          f"|  F1*={summary['f1_best_mean']:.4f}±{summary['f1_best_std']:.4f} "
          f"@thr={summary['best_threshold_mean']:.3f}  "
          f"({elapsed:.0f}s)")

    # Train final deployment model on the same downsampled distribution
    # as the CV folds (all positives + 10× negatives) for consistency.
    X_final, y_final = downsample_train(X, y, NEG_TO_POS_RATIO, seed=RANDOM_STATE)
    final_model = get_model(model_name, feature_name)
    final_model.fit(X_final, y_final)
    model_path = MODELS_DIR / f"{slug(model_name)}_{feature_name}.pkl"
    joblib.dump(final_model, model_path)
    print(f"  → Saved final model: {model_path}  "
          f"(trained on {X_final.shape[0]:,} downsampled rows)")

    return out_dir


def pvalue_str(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return "<0.01"
    elif p < 0.05:
        return "<0.05"
    return f"{p:.3f}"


def summarize(feature_name: str, reference_model: str = "Random Forest"):
    """Load all completed experiment summaries and print a Table-2-style comparison."""
    rows = []
    for entry in MODEL_REGISTRY:
        model_name = entry[0]
        out_dir = experiment_dir(model_name, feature_name)
        summary_path = out_dir / "summary.json"
        fold_path = out_dir / "fold_metrics.csv"
        if not summary_path.exists():
            continue
        s = json.loads(summary_path.read_text())

        fold_vals = {"f1": [], "auroc": [], "f1_best": []}
        with open(fold_path) as f:
            for row in csv.DictReader(f):
                fold_vals["f1"].append(float(row["f1"]))
                fold_vals["auroc"].append(float(row["auroc"]))
                # f1_best may be absent in old experiments (pre-threshold-sweep)
                if row.get("f1_best"):
                    fold_vals["f1_best"].append(float(row["f1_best"]))

        rows.append((model_name, s, fold_vals))

    if not rows:
        print("No completed experiments found.")
        return

    # Find reference
    ref_vals = None
    for name, s, vals in rows:
        if name == reference_model:
            ref_vals = vals
            break

    print(f"\n{'='*128}")
    print(f"  Feature set: {feature_name}  |  Reference model (for p-values): {reference_model}")
    print(f"{'='*128}")
    header = (f"{'Model':<22}  {'Precision':^20}  {'Recall':^20}  {'F-1':^20}  "
              f"{'AUROC':^20}  {'F1* (best thr)':^20}  {'p(F1)':>8}  {'p(AUROC)':>10}")
    print(header)
    print("-" * 128)

    for model_name, s, vals in rows:
        prec = f"{s['precision_mean']:.4f} ± {s['precision_std']:.4f}"
        rec  = f"{s['recall_mean']:.4f} ± {s['recall_std']:.4f}"
        f1   = f"{s['f1_mean']:.4f} ± {s['f1_std']:.4f}"
        auroc = f"{s['auroc_mean']:.4f} ± {s['auroc_std']:.4f}"
        if "f1_best_mean" in s:
            f1b = f"{s['f1_best_mean']:.4f} ± {s['f1_best_std']:.4f}"
        else:
            f1b = "—"

        if ref_vals is None or model_name == reference_model:
            pf1 = "—"
            pau = "—"
        else:
            _, pv_f1   = ttest_rel(vals["f1"],   ref_vals["f1"])
            _, pv_auroc = ttest_rel(vals["auroc"], ref_vals["auroc"])
            pf1 = pvalue_str(pv_f1)
            pau = pvalue_str(pv_auroc)

        print(f"{model_name:<22}  {prec:^20}  {rec:^20}  {f1:^20}  {auroc:^20}  "
              f"{f1b:^20}  {pf1:>8}  {pau:>10}")

    print("=" * 128)

    # Save combined CSV
    out_path = Path(__file__).parent.parent / "results" / f"table2_{feature_name}.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model",
            "Precision_mean", "Precision_std",
            "Recall_mean", "Recall_std",
            "F1_mean", "F1_std",
            "AUROC_mean", "AUROC_std",
            "F1_best_mean", "F1_best_std",
            "Precision_at_best_mean", "Recall_at_best_mean",
            "best_threshold_mean",
            "pvalue_F1", "pvalue_AUROC",
        ])
        for model_name, s, vals in rows:
            if ref_vals is None or model_name == reference_model:
                pf1 = pau = "—"
            else:
                _, pv_f1    = ttest_rel(vals["f1"],   ref_vals["f1"])
                _, pv_auroc = ttest_rel(vals["auroc"], ref_vals["auroc"])
                pf1 = pvalue_str(pv_f1)
                pau = pvalue_str(pv_auroc)
            writer.writerow([
                model_name,
                s["precision_mean"], s["precision_std"],
                s["recall_mean"],    s["recall_std"],
                s["f1_mean"],        s["f1_std"],
                s["auroc_mean"],     s["auroc_std"],
                s.get("f1_best_mean", "—"),       s.get("f1_best_std", "—"),
                s.get("precision_at_best_mean", "—"),
                s.get("recall_at_best_mean", "—"),
                s.get("best_threshold_mean", "—"),
                pf1, pau,
            ])
    print(f"\nSaved combined table to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="top100",
                        help="Feature set name. Matrix loaded from "
                             "data/matrix_{name}.npz, sequence from "
                             "data/sequences_{name}.npy.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help=f"Model name. Choices: {get_model_names()}")
    group.add_argument("--all", action="store_true", help="Run all models in the registry")
    group.add_argument("--summarize", action="store_true",
                       help="Print summary table from completed experiments")
    args = parser.parse_args()

    if args.summarize:
        summarize(args.features)
        return

    if args.all:
        # Different models may need different data sources; reload per group.
        cache: dict[str, tuple] = {}
        for model_name, _, data_type in MODEL_REGISTRY:
            if data_type not in cache:
                cache[data_type] = load_data(args.features, data_type)
                X, y = cache[data_type]
                print(f"Loaded {data_type}: shape={X.shape}, "
                      f"positive_rate={y.mean():.4f}", flush=True)
            X, y = cache[data_type]
            run_model(model_name, args.features, X, y)
        summarize(args.features)
    else:
        data_type = get_data_type(args.model)
        X, y = load_data(args.features, data_type)
        print(f"Loaded {data_type}: shape={X.shape}, "
              f"positive_rate={y.mean():.4f}", flush=True)
        run_model(args.model, args.features, X, y)


if __name__ == "__main__":
    main()
