"""
Confusion-matrix grid plot at a calibrated threshold.

Reads experiments/{slug}_{fs}/fold_predictions.npz (written by 05_calibration_eval.py)
and aggregates the 10-fold per-patient predictions into a single confusion matrix
per model. Default threshold strategy is prevalence-matched (clinical operating
point); --threshold f1_opt or default(0.5) also supported.

Layout: 2x4 grid (8 models at top300 by default).
Color: row-normalized (each row sums to 1) so diagonal intensity shows
per-class accuracy. Annotations show raw counts and the row-percentage.

Outputs:
  results/confusion_matrix_{fs}_{strategy}.png

Usage:
  python src/plot_confusion_matrix.py                              # top300, prevalence-matched
  python src/plot_confusion_matrix.py --threshold f1_opt
  python src/plot_confusion_matrix.py --threshold default --features top200
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
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).parent))
from model_registry import MODEL_REGISTRY

ROOT = Path(__file__).parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def threshold_for(strategy: str, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if strategy == "default":
        return 0.5
    if strategy == "f1_opt":
        p, r, t = precision_recall_curve(y_true, y_score)
        p, r = p[:-1], r[:-1]
        f1s = 2 * p * r / np.clip(p + r, 1e-12, None)
        if len(f1s) == 0:
            return 0.5
        return float(t[int(np.argmax(f1s))])
    if strategy == "prevalence":
        prev = float(y_true.mean())
        return float(np.quantile(y_score, 1.0 - prev))
    raise ValueError(strategy)


STRAT_NICE = {
    "default": "Default (thr=0.5)",
    "f1_opt": "F1-optimal threshold",
    "prevalence": "Prevalence-matched threshold",
}


def confusion_per_model(model_name: str, fs: str, strategy: str):
    """Aggregate the 10-fold predictions into a single confusion matrix
    (each patient appears in exactly one test fold, so the sum covers the
    whole dataset under the CV protocol)."""
    p = EXP / f"{slug(model_name)}_{fs}" / "fold_predictions.npz"
    if not p.exists():
        return None
    z = np.load(p)
    folds = z["fold"]
    y_true = z["y_true"].astype(np.int8)
    y_score = z["y_score"].astype(np.float32)

    tn = fp = fn = tp = 0
    for k in np.unique(folds):
        sel = folds == k
        yt = y_true[sel]
        ys = y_score[sel]
        thr = threshold_for(strategy, yt, ys)
        yp = (ys >= thr).astype(np.int8)
        tn += int(((yt == 0) & (yp == 0)).sum())
        fp += int(((yt == 0) & (yp == 1)).sum())
        fn += int(((yt == 1) & (yp == 0)).sum())
        tp += int(((yt == 1) & (yp == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def plot_grid(fs: str, strategy: str):
    mats = []
    for entry in MODEL_REGISTRY:
        name = entry[0]
        m = confusion_per_model(name, fs, strategy)
        if m is not None:
            mats.append((name, m))

    if not mats:
        print(f"No fold_predictions.npz found for {fs}.")
        return

    ncols = 4
    nrows = (len(mats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows),
                             squeeze=False)

    for i, (name, cm) in enumerate(mats):
        ax = axes[i // ncols][i % ncols]
        # row-normalize for color (per-true-class)
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
        for r in range(2):
            for c in range(2):
                count = cm[r, c]
                pct = norm[r, c] * 100
                color = "white" if norm[r, c] > 0.5 else "black"
                ax.text(c, r, f"{_fmt_count(count)}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=10, color=color)
        # compute headline metrics from cm
        tn, fp, fn, tp = cm.ravel()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        ax.set_title(f"{name}\nP={prec:.3f}  R={rec:.3f}  F1={f1:.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred neg", "Pred pos"], fontsize=9)
        ax.set_yticklabels(["True neg", "True pos"], fontsize=9)
        ax.set_xlabel(""); ax.set_ylabel("")

    for j in range(len(mats), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Confusion matrices — {fs}, {STRAT_NICE[strategy]}\n"
                 f"(sum over all 10 CV test folds; color = row-normalized)",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT / f"confusion_matrix_{fs}_{strategy}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  ({len(mats)} models)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="top300",
                        choices=["top50", "top100", "top150", "top200", "top300"])
    parser.add_argument("--threshold", default="prevalence",
                        choices=["default", "f1_opt", "prevalence"])
    args = parser.parse_args()
    plot_grid(args.features, args.threshold)


if __name__ == "__main__":
    main()
