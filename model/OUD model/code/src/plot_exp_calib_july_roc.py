"""
ROC curves for the exp_calib_july pilot: Logistic Regression and Transformer,
each on the three curation families (Kelly / Rick / Full). Color encodes
feature set (matches plot_roc_curves_six_experiments.py's palette). Reads
from experiments/exp_calib_july/ (single train/calibration/test split, not
10-fold CV) and marks two operating points on every curve:

  o  default (0.5)     — raw score thresholded at 0.5 (the naive operating
                         point, before any calibration/threshold work)
  *  true prevalence   — the prevalence_matched operating point (predicted
                         positive rate ~= calibration split's own ~1.16%
                         prevalence). Same point on the curve regardless of
                         raw/isotonic/platt (monotonic transforms preserve
                         rank), so it's read off isotonic's threshold
                         (lowest Brier of the three methods) and applied to
                         the isotonic score.

Produces TWO separate figures, distinct filenames, neither overwrites the
other:
  roc_curve_combined.png/.pdf   — one axes, 6 lines. Rick and Full are close
                                  enough in AUROC that color+linestyle (the
                                  original design) made the 6 lines hard to
                                  tell apart where they overlap, so this
                                  figure instead uses hue=feature set (3
                                  well-separated hues) x shade=model (dark=LR,
                                  light=Transformer), all solid strokes — no
                                  dashed segments competing for attention.
  roc_curve_by_model.png/.pdf   — two panels, LR left / Transformer right
                                  (color=feature set; unaffected by the above,
                                  each panel only has 3 lines so the original
                                  single-hue-per-feature-set palette is fine)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).parent.parent
EXP = ROOT / "experiments" / "exp_calib_july"
OUT = ROOT / "results" / "exp_calib_july" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    ("Logistic Regression", "Kelly", "logistic_regression_kelly"),
    ("Logistic Regression", "Rick", "logistic_regression_rick_top251"),
    ("Logistic Regression", "Full", "logistic_regression_top300"),
    ("Transformer", "Kelly", "transformer_kelly"),
    ("Transformer", "Rick", "transformer_rick_top251"),
    ("Transformer", "Full", "transformer_top300"),
]

# Same palette as plot_roc_curves_six_experiments.py, for visual continuity
# across the paper's other Kelly/Rick/Full figures.
COLORS = {
    "Kelly": "#1f77b4",
    "Rick": "#2ca02c",
    "Full": "#4b5563",
}
LINESTYLES = {
    "Logistic Regression": "-",
    "Transformer": "--",
}

# roc_curve_combined only: hue (feature set) x shade (model), all solid.
# Kelly=blue, Rick=green, Full=orange (swapped from COLORS' gray — gray read
# too close to the gridlines/"Random" diagonal once 6 lines were on one
# axes). Dark step = Logistic Regression, light step = Transformer, so model
# identity survives even where Rick and Full nearly overlap (their AUROCs
# are within ~0.004 of each other). Steps taken from ColorBrewer
# Blues/Greens/Oranges (perceptually-tested sequential ramps).
COMBINED_COLORS = {
    ("Kelly", "Logistic Regression"): "#08519c",
    ("Kelly", "Transformer"):         "#6baed6",
    ("Rick", "Logistic Regression"):  "#238b45",
    ("Rick", "Transformer"):          "#74c476",
    ("Full", "Logistic Regression"):  "#d94801",
    ("Full", "Transformer"):          "#fd8d3c",
}

MARKER_HANDLES = [
    plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666",
               markeredgecolor="white", markersize=14, label="default (0.5)"),
    plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#666666",
               markeredgecolor="white", markersize=22, label="true prevalence (~1.16%)"),
]


def confusion_rates(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> tuple[float, float]:
    """Return (FPR, TPR) at score >= thr."""
    y_pred = y_score >= thr
    pos = y_true == 1
    neg = ~pos
    tpr = float(y_pred[pos].mean()) if pos.any() else 0.0
    fpr = float(y_pred[neg].mean()) if neg.any() else 0.0
    return fpr, tpr


def load_experiment(slug: str):
    z = np.load(EXP / slug / "predictions.npz")
    y_true = z["y_true"].astype(np.int8)
    raw_score = z["raw_score"].astype(np.float32)
    isotonic_score = z["isotonic_score"].astype(np.float32)
    summary = json.loads((EXP / slug / "summary_calib_split.json").read_text())
    return y_true, raw_score, isotonic_score, summary


def compute_curves():
    """Load every (model, feature_set) once and compute everything needed
    to draw either figure, so both figures are built from identical numbers."""
    curves = []
    rows = []
    for model, feature_set, slug in EXPERIMENTS:
        pred_path = EXP / slug / "predictions.npz"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        y_true, raw_score, isotonic_score, summary = load_experiment(slug)

        # ROC curve/AUROC are calibration-invariant (monotonic transforms
        # preserve rank), so the curve is drawn from raw_score regardless.
        fpr, tpr, _ = roc_curve(y_true, raw_score)
        auc = roc_auc_score(y_true, raw_score)

        # default(0.5): kept on the RAW score deliberately. Under isotonic/
        # platt the calibrated probability almost never exceeds 0.5 at this
        # prevalence (platt predicts exactly zero positives in every one of
        # the 6 combos), so those points would collapse onto/near the origin
        # and be visually indistinguishable — raw's point is the informative
        # one here (it's the "naive, uncalibrated" operating point).
        thr_default = 0.5
        fpr_d, tpr_d = confusion_rates(y_true, raw_score, thr_default)

        # true prevalence: read off isotonic's own threshold (isotonic had
        # the lowest Brier score of the three methods in every combo) and
        # applied to the isotonic score. This lands on the exact same
        # (FPR, TPR) point as raw/platt would (prevalence_matched is a
        # rank-based top-k selection, invariant to monotonic rescaling).
        thr_prev = summary["methods"]["isotonic"]["prevalence_matched"]["threshold"]
        fpr_p, tpr_p = confusion_rates(y_true, isotonic_score, thr_prev)

        curves.append({
            "model": model, "feature_set": feature_set, "auc": auc,
            "fpr": fpr, "tpr": tpr,
            "default_point": (fpr_d, tpr_d), "prevalence_point": (fpr_p, tpr_p),
        })
        rows.append({
            "model": model, "feature_set": feature_set, "n": len(y_true),
            "prevalence_test": f"{float(y_true.mean()):.6f}", "auroc": f"{auc:.6f}",
            "default_threshold": thr_default, "default_fpr": f"{fpr_d:.6f}", "default_tpr": f"{tpr_d:.6f}",
            "prevalence_threshold": f"{thr_prev:.6f}", "prevalence_fpr": f"{fpr_p:.6f}", "prevalence_tpr": f"{tpr_p:.6f}",
        })
    return curves, rows


def plot_combined(curves):
    # Sized/fonted for slide embedding — sig. larger than a print/PDF figure.
    fig, ax = plt.subplots(figsize=(13.5, 13.0))

    for c in curves:
        color = COMBINED_COLORS[(c["feature_set"], c["model"])]
        ax.plot(c["fpr"], c["tpr"], color=color, lw=2.8,
                 label=f"{c['model']} · {c['feature_set']} (AUROC={c['auc']:.3f})")
        ax.scatter(*c["default_point"], marker="o", s=420, facecolor=color,
                   edgecolor="white", linewidth=2.6, zorder=5)
        ax.scatter(*c["prevalence_point"], marker="*", s=1300, facecolor=color,
                   edgecolor="white", linewidth=2.0, zorder=5)

    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1.4, label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=26)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=26)
    ax.set_title("ROC Curve", fontsize=34, fontweight="bold", pad=18)
    ax.tick_params(axis="both", labelsize=21)
    ax.grid(True, alpha=0.28)
    ax.set_aspect("equal", adjustable="box")

    curve_legend = ax.legend(loc="lower right", fontsize=23, frameon=True)
    ax.add_artist(curve_legend)
    # Larger proxy markers than MARKER_HANDLES (used by plot_by_model), to
    # match this figure's bigger on-plot markers — kept local so it doesn't
    # affect the by-model figure's legend.
    marker_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666",
                   markeredgecolor="white", markersize=28, label="default (0.5)"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#666666",
                   markeredgecolor="white", markersize=46, label="true prevalence (~1.16%)"),
    ]
    fig.legend(handles=marker_handles, loc="lower center", ncol=2,
               fontsize=27, frameon=False, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_png = OUT / "roc_curve_combined.png"
    out_pdf = OUT / "roc_curve_combined.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(out_png)
    print(out_pdf)


def plot_by_model(curves):
    fig, axes = plt.subplots(1, 2, figsize=(19.0, 10.6), sharey=True)

    for c in curves:
        ax = axes[0] if c["model"] == "Logistic Regression" else axes[1]
        color = COLORS[c["feature_set"]]
        ax.plot(c["fpr"], c["tpr"], color=color, lw=2.2,
                 label=f"{c['feature_set']} (AUROC={c['auc']:.3f})")
        ax.scatter(*c["default_point"], marker="o", s=140, facecolor=color,
                   edgecolor="white", linewidth=1.6, zorder=5)
        ax.scatter(*c["prevalence_point"], marker="*", s=380, facecolor=color,
                   edgecolor="white", linewidth=1.3, zorder=5)

    for ax, title in zip(axes, ["Logistic Regression", "Transformer"]):
        ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1.0, label="Random")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=17)
        ax.set_title(title, fontsize=19, fontweight="bold")
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, alpha=0.28)
        curve_legend = ax.legend(loc="lower right", fontsize=13, frameon=True)
        ax.add_artist(curve_legend)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=17)

    fig.suptitle("ROC Curve", fontsize=22, fontweight="bold")
    fig.legend(handles=MARKER_HANDLES, loc="lower center", ncol=2,
               fontsize=16, frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    out_png = OUT / "roc_curve_by_model.png"
    out_pdf = OUT / "roc_curve_by_model.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(out_png)
    print(out_pdf)


def plot_single(curves, model: str, feature_set: str, out_stem: str, color: str):
    """One line only — e.g. Transformer · Full. Same slide-sized fonts and
    the same two operating points as the other figures."""
    c = next(c for c in curves if c["model"] == model and c["feature_set"] == feature_set)

    fig, ax = plt.subplots(figsize=(13.5, 13.0))

    ax.plot(c["fpr"], c["tpr"], color=color, lw=3.4,
             label=f"{model} · {feature_set} (AUROC={c['auc']:.3f})")
    ax.scatter(*c["default_point"], marker="o", s=460, facecolor=color,
               edgecolor="white", linewidth=2.6, zorder=5)
    ax.scatter(*c["prevalence_point"], marker="*", s=1400, facecolor=color,
               edgecolor="white", linewidth=2.0, zorder=5)

    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1.4, label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=26)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=26)
    ax.set_title("ROC Curve", fontsize=34, fontweight="bold", pad=18)
    ax.tick_params(axis="both", labelsize=21)
    ax.grid(True, alpha=0.28)
    ax.set_aspect("equal", adjustable="box")

    curve_legend = ax.legend(loc="lower right", fontsize=24, frameon=True)
    ax.add_artist(curve_legend)
    marker_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666",
                   markeredgecolor="white", markersize=28, label="default (0.5)"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#666666",
                   markeredgecolor="white", markersize=46, label="true prevalence (~1.16%)"),
    ]
    fig.legend(handles=marker_handles, loc="lower center", ncol=2,
               fontsize=27, frameon=False, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_png = OUT / f"{out_stem}.png"
    out_pdf = OUT / f"{out_stem}.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(out_png)
    print(out_pdf)


def main():
    curves, rows = compute_curves()

    plot_combined(curves)
    plot_by_model(curves)
    plot_single(curves, "Transformer", "Full", "roc_curve_transformer_full",
                color=COMBINED_COLORS[("Full", "Transformer")])

    out_csv = OUT / "roc_curve_operating_points.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_csv)


if __name__ == "__main__":
    main()
