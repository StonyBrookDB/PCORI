"""
Generate comparison plots from completed experiments.

Reads experiments/{model_slug}_{feature_set}/summary.json and produces:
  results/plot_auroc_vs_features.png    AUROC vs feature set size, one line per model
  results/plot_f1_vs_features.png       F1    vs feature set size, one line per model
  results/plot_metrics_grid.png         2×2 grid: precision / recall / F1 / AUROC

Usage:
  python src/plot_results.py
  python src/plot_results.py --models "Logistic Regression" "Decision Tree" "Random Forest" "DNN"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model_registry import MODEL_REGISTRY

ROOT = Path(__file__).parent.parent
EXP_DIR = ROOT / "experiments"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

FEATURE_SETS = ["top50", "top100", "top150", "top200", "top300"]
FEATURE_SIZES = {"top50": 50, "top100": 100, "top150": 150, "top200": 200, "top300": 299}

# Distinct colors per model (matplotlib 'tab10' is colorblind-friendly enough)
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
MARKERS = {
    "Logistic Regression": "o",
    "Decision Tree":       "s",
    "Random Forest":       "^",
    "DNN":                 "D",
    "LSTM":                "v",
    "Bi-LSTM":             "P",
    "Attention":           "X",
    "Transformer":         "*",
}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_summary(model_name: str, fs: str) -> dict | None:
    path = EXP_DIR / f"{slug(model_name)}_{fs}" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def collect(model_names: list[str]) -> dict[str, dict[str, dict]]:
    """Returns {model_name: {feature_set: summary_dict}} for completed runs only."""
    out: dict[str, dict[str, dict]] = {}
    for m in model_names:
        runs = {}
        for fs in FEATURE_SETS:
            s = load_summary(m, fs)
            if s is not None:
                runs[fs] = s
        if runs:
            out[m] = runs
    return out


def _plot_metric(ax, data, metric: str, ylabel: str, title: str):
    x_all = [FEATURE_SIZES[fs] for fs in FEATURE_SETS]
    for model_name, runs in data.items():
        # Build x/y arrays, skipping missing feature sets or missing fields
        xs, ys, ys_err = [], [], []
        for fs in FEATURE_SETS:
            if fs not in runs:
                continue
            if f"{metric}_mean" not in runs[fs]:
                # Old summary.json without this metric (e.g. f1_best pre-sweep)
                continue
            xs.append(FEATURE_SIZES[fs])
            ys.append(runs[fs][f"{metric}_mean"])
            ys_err.append(runs[fs][f"{metric}_std"])
        if not xs:
            continue
        ax.errorbar(
            xs, ys, yerr=ys_err,
            label=model_name,
            color=COLORS.get(model_name, None),
            marker=MARKERS.get(model_name, "o"),
            markersize=7, linewidth=1.6, capsize=3,
        )
    ax.set_xlabel("Feature set size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_all)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def plot_single(data: dict, metric: str, out_path: Path, ylabel: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_metric(ax, data, metric, ylabel, title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_grid(data: dict, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    panels = [
        ("precision", "Precision", "Precision @ threshold 0.5"),
        ("recall",    "Recall",    "Recall @ threshold 0.5"),
        ("f1",        "F1",        "F1 @ threshold 0.5"),
        ("auroc",     "AUROC",     "AUROC (threshold-independent)"),
        ("f1_best",   "F1*",       "F1 @ F1-optimal threshold per fold"),
        ("best_threshold", "Threshold", "F1-optimal threshold (mean over folds)"),
    ]
    for ax, (metric, ylabel, title) in zip(axes.flat, panels):
        _plot_metric(ax, data, metric, ylabel, title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=[name for name, _, _ in MODEL_REGISTRY],
        help="Models to include in the plot (default: all registered).",
    )
    parser.add_argument(
        "--suffix", default="",
        help="Optional filename suffix, e.g. '_baselines'.",
    )
    args = parser.parse_args()

    data = collect(args.models)
    if not data:
        print("No completed experiments found.")
        return

    print(f"Plotting {len(data)} models across up to {len(FEATURE_SETS)} feature sets:")
    for m, runs in data.items():
        feats = ", ".join(runs.keys())
        print(f"  - {m}: {feats}")

    suffix = args.suffix
    plot_single(
        data, "auroc",
        OUT_DIR / f"plot_auroc_vs_features{suffix}.png",
        "AUROC", "AUROC vs feature set size",
    )
    plot_single(
        data, "f1",
        OUT_DIR / f"plot_f1_vs_features{suffix}.png",
        "F1 @ threshold 0.5", "F1 @ threshold 0.5 vs feature set size",
    )
    plot_single(
        data, "f1_best",
        OUT_DIR / f"plot_f1_best_vs_features{suffix}.png",
        "F1*  (F1-optimal threshold)",
        "F1 at F1-optimal threshold vs feature set size",
    )
    plot_grid(data, OUT_DIR / f"plot_metrics_grid{suffix}.png")


if __name__ == "__main__":
    main()
