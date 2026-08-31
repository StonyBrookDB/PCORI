"""
Plot ROC curves for the paper models from saved per-fold predictions.

Reads experiments/{slug}_{fs}/fold_predictions.npz (written by 05_calibration_eval.py),
concatenates all 10 folds' (y_true, y_score) per (model, feature_set) — since each
patient appears in exactly one test fold, this covers the full dataset under CV —
and plots ROC with AUC.

Outputs:
  results/roc_curve_{fs}.png       — overlay of all requested models

Usage:
  python src/plot_roc_curve.py                                           # LR + Transformer @ top300
  python src/plot_roc_curve.py --features top200
  python src/plot_roc_curve.py --models "Logistic Regression" Transformer DNN Attention
  python src/plot_roc_curve.py --all-baselines                           # 8 models at top300
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from model_registry import MODEL_REGISTRY

ROOT = Path(__file__).parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "results"

DEFAULT_MODELS = ["Logistic Regression", "Transformer"]
ALL_MODELS = [name for name, _, _ in MODEL_REGISTRY]

# Colors picked to match other paper figures
COLORS = {
    "Logistic Regression": "#1f77b4",
    "Decision Tree":       "#ff7f0e",
    "Random Forest":       "#2ca02c",
    "DNN":                 "#d62728",
    "LSTM":                "#9467bd",
    "Bi-LSTM":             "#8c564b",
    "Attention":           "#e377c2",
    "Transformer":         "#7f7f7f",
}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_predictions(model: str, fs: str):
    p = EXP / f"{slug(model)}_{fs}" / "fold_predictions.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["y_true"].astype(np.int8), z["y_score"].astype(np.float32)


def plot_roc(models: list[str], fs: str, suffix: str = ""):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1, label="Random (AUC=0.5)")

    plotted = []
    for m in models:
        d = load_predictions(m, fs)
        if d is None:
            print(f"  - {m} @ {fs}: no fold_predictions.npz, skipping")
            continue
        y_true, y_score = d
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, lw=1.8, color=COLORS.get(m, None),
                label=f"{m}  (AUC = {auc:.4f})")
        plotted.append(m)

    if not plotted:
        print("No predictions available — nothing to plot.")
        plt.close(fig)
        return

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC curve — {fs}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    out = OUT / f"roc_curve_{fs}{suffix}.png"
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  ({len(plotted)} models)")


# Nicer legend labels for feature-set names
FEATURE_SET_LABELS = {
    "kelly": "Kelly limited (48)",
    "rick_top50": "Rick top-50",
    "rick_top100": "Rick top-100",
    "rick_top150": "Rick top-150",
    "rick_top200": "Rick top-200",
    "rick_top251": "Rick expanded (251)",
    "top50": "top-50",
    "top100": "top-100",
    "top150": "top-150",
    "top200": "top-200",
    "top300": "top-300 (inclusive)",
}

# Distinct color ramp for feature-set overlays (single model, multiple curves)
_FS_PALETTE = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef"]


def plot_roc_across_features(model: str, feature_sets: list[str], suffix: str = ""):
    """Overlay ROC curves for ONE model across several feature sets."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1, label="Random (AUC=0.5)")

    plotted = []
    for i, fs in enumerate(feature_sets):
        d = load_predictions(model, fs)
        if d is None:
            print(f"  - {model} @ {fs}: no fold_predictions.npz, skipping")
            continue
        y_true, y_score = d
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        color = _FS_PALETTE[i % len(_FS_PALETTE)]
        label = FEATURE_SET_LABELS.get(fs, fs)
        ax.plot(fpr, tpr, lw=1.8, color=color, label=f"{label}  (AUC = {auc:.4f})")
        plotted.append(fs)

    if not plotted:
        print("No predictions available — nothing to plot.")
        plt.close(fig)
        return

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC curve — {model} across feature sets")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    out = OUT / f"roc_curve_{slug(model)}_by_features{suffix}.png"
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  ({len(plotted)} feature sets)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="top300",
                        help="Feature set name (default top300)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to overlay (default: LR + Transformer)")
    parser.add_argument("--all-baselines", action="store_true",
                        help="Plot all 8 registered models")
    parser.add_argument("--by-features", nargs="+", default=None,
                        help="Overlay ONE model across multiple feature sets "
                             "instead of multiple models on one feature set. "
                             "Use with --models (single model).")
    parser.add_argument("--suffix", default="",
                        help="Optional output filename suffix")
    args = parser.parse_args()

    if args.by_features:
        model = (args.models or DEFAULT_MODELS)[0]
        plot_roc_across_features(model, args.by_features, args.suffix)
        return

    if args.all_baselines:
        models = ALL_MODELS
        suffix = args.suffix or "_all"
    elif args.models:
        models = args.models
        suffix = args.suffix
    else:
        models = DEFAULT_MODELS
        suffix = args.suffix

    plot_roc(models, args.features, suffix)


if __name__ == "__main__":
    main()
