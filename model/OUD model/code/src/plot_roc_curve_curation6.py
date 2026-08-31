"""
Combined ROC curve: Logistic Regression + Transformer, each at 3 feature-curation
levels (Inclusive top-300, Rick expanded 251, Kelly limited 48) -> 6 lines total.

Each curve is annotated with two operating points:
  - the threshold = 0.5 point (circle marker)
  - the prevalence-matched point, i.e. the threshold where predicted-positive
    rate equals the true (natural) prevalence of the full dataset (triangle marker)

Read-only: only reads existing experiments/*/fold_predictions.npz. Writes only to
results/figuresHW15/ -- does not touch any existing results file.

Usage:
  python src/plot_roc_curve_curation6.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from plot_roc_curve import load_predictions  # reuse existing loader

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "results" / "figuresHW15"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMBOS = [
    ("Logistic Regression", "top300", "Inclusive (299)"),
    ("Logistic Regression", "rick_top251", "Rick expanded (251)"),
    ("Logistic Regression", "kelly", "Kelly limited (48)"),
    ("Transformer", "top300", "Inclusive (299)"),
    ("Transformer", "rick_top251", "Rick expanded (251)"),
    ("Transformer", "kelly", "Kelly limited (48)"),
]

# Sequential ramps per model identity (darker = more features), consistent with
# the blue/green identity already used for LR/Transformer elsewhere in this project.
LR_SHADES = {"top300": "#08306b", "rick_top251": "#2171b5", "kelly": "#6baed6"}
TF_SHADES = {"top300": "#00441b", "rick_top251": "#238b45", "kelly": "#74c476"}


def shade(model: str, fs: str) -> str:
    return LR_SHADES[fs] if model == "Logistic Regression" else TF_SHADES[fs]


def operating_points(y_true: np.ndarray, y_score: np.ndarray):
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    true_prev = n_pos / len(y_true)

    pred05 = (y_score >= 0.5)
    tpr05 = ((pred05) & (y_true == 1)).sum() / n_pos
    fpr05 = ((pred05) & (y_true == 0)).sum() / n_neg

    thr_prev = np.quantile(y_score, 1 - true_prev)
    pred_prev = (y_score >= thr_prev)
    tpr_prev = ((pred_prev) & (y_true == 1)).sum() / n_pos
    fpr_prev = ((pred_prev) & (y_true == 0)).sum() / n_neg

    return (fpr05, tpr05), (fpr_prev, tpr_prev), true_prev


plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
})


def main():
    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1.5, label="Random (AUC=0.5)")

    true_prev_ref = None
    for model, fs, label in COMBOS:
        d = load_predictions(model, fs)
        if d is None:
            print(f"  - {model} @ {fs}: no fold_predictions.npz, skipping")
            continue
        y_true, y_score = d
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        color = shade(model, fs)
        ax.plot(fpr, tpr, lw=3, color=color,
                 label=f"{model} — {label}  (AUC={auc:.3f})")

        (fpr05, tpr05), (fpr_prev, tpr_prev), true_prev = operating_points(y_true, y_score)
        true_prev_ref = true_prev
        ax.plot(fpr05, tpr05, marker="o", ms=20, mfc=color, mec="white", mew=2.2, zorder=5)
        ax.plot(fpr_prev, tpr_prev, marker="^", ms=23, mfc=color, mec="white", mew=2.2, zorder=5)

    marker_legend = [
        Line2D([0], [0], marker="o", linestyle="none", mfc="#555555", mec="white",
               mew=1.6, ms=18, label="Threshold = 0.5"),
        Line2D([0], [0], marker="^", linestyle="none", mfc="#555555", mec="white",
               mew=1.6, ms=21,
               label=f"Prevalence-matched (true prevalence = {true_prev_ref:.2%})"
               if true_prev_ref else "Prevalence-matched"),
    ]

    ax.set_xlim(-0.035, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC — Logistic Regression vs Transformer\nacross feature curation (Inclusive / Rick / Kelly)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(width=1.2, length=6)

    leg1 = ax.legend(loc="lower right", fontsize=13, title="Model — feature set",
                      title_fontsize=14, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=marker_legend, loc="upper left", fontsize=13, framealpha=0.95)

    out = OUT_DIR / "roc_curve_curation_6lines.png"
    plt.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
