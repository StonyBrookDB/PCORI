"""
Calibration-quality figures for the exp_calib_july pilot: Logistic Regression
and Transformer, each on Kelly / Rick / Full List 299 — 2x3 grid (rows =
model, cols = feature set), reading from experiments/exp_calib_july/.

Produces TWO figures:

  calibration_reliability_grid.png/.pdf
      Reliability diagram (mean predicted probability vs. observed positive
      fraction) for raw / isotonic / platt, log-log axes (values span three
      orders of magnitude at ~1.16% prevalence — a linear axis would collapse
      the calibrated curves into an unreadable sliver near the origin).
      Data is the reliability_curve_test block already computed and stored
      in summary_calib_split.json (quantile-binned on the test split) — not
      recomputed here. A raw curve sitting well ABOVE the diagonal means the
      model over-predicts risk almost everywhere (expected: class_weight=
      "balanced"/1:10 downsampling pushes scores toward 0.5 regardless of the
      true ~1.16% prior); isotonic/platt should track the diagonal closely.

  calibration_mapping_grid.png/.pdf
      The actual fitted mapping, raw score (x) -> calibrated score (y), for
      isotonic and platt. Read directly off predictions.npz's (raw_score,
      isotonic_score, platt_score) triples for the test split — no calibrator
      refit needed, since the mapping is a deterministic function applied
      elementwise (identical raw_score always produces identical calibrated
      score). Points are sorted by raw_score and subsampled for a clean line
      (per-point plotting of 1.77M rows would be slow and no more informative
      — the mapping has no noise to show). A y=x dashed reference line marks
      "no change", so the extent of the recalibration is visible directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
EXP = ROOT / "experiments" / "exp_calib_july"
OUT = ROOT / "results" / "exp_calib_july" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["Logistic Regression", "Transformer"]
FEATURE_SETS = ["Kelly", "Rick", "Full List 299"]
SLUGS = {
    ("Logistic Regression", "Kelly"): "logistic_regression_kelly",
    ("Logistic Regression", "Rick"): "logistic_regression_rick_top251",
    ("Logistic Regression", "Full List 299"): "logistic_regression_top300",
    ("Transformer", "Kelly"): "transformer_kelly",
    ("Transformer", "Rick"): "transformer_rick_top251",
    ("Transformer", "Full List 299"): "transformer_top300",
}

# Distinct from the Kelly/Rick/Full palette used in the ROC figures (those
# colors mean FEATURE SET there; here color means CALIBRATION METHOD, so a
# different palette avoids readers assuming the same color means the same
# thing across figures).
METHOD_COLORS = {
    "raw": "#999999",
    "isotonic": "#2a9d8f",
    "platt": "#e76f51",
}
METHOD_LABELS = {
    "raw": "Raw (uncalibrated)",
    "isotonic": "Isotonic",
    "platt": "Platt",
}
SUBSAMPLE_POINTS = 3000


def load_summary(slug: str) -> dict:
    return json.loads((EXP / slug / "summary_calib_split.json").read_text())


def load_predictions(slug: str):
    z = np.load(EXP / slug / "predictions.npz")
    return z["raw_score"].astype(np.float32), z["isotonic_score"].astype(np.float32), z["platt_score"].astype(np.float32)


def plot_reliability_grid():
    # Compact but large-font: width kept wide enough for 3 columns of titles/
    # labels not to collide; height trimmed to remove dead vertical space.
    fig, axes = plt.subplots(2, 3, figsize=(25.0, 13.5), gridspec_kw={"hspace": 0.32, "wspace": 0.28})

    for i, model in enumerate(MODELS):
        for j, fs in enumerate(FEATURE_SETS):
            ax = axes[i, j]
            slug = SLUGS[(model, fs)]
            summary = load_summary(slug)

            for method in ["raw", "isotonic", "platt"]:
                curve = summary["reliability_curve_test"][method]
                x = [r["mean_predicted"] for r in curve]
                y = [r["frac_positive"] for r in curve]
                ax.plot(x, y, "o-", color=METHOD_COLORS[method], lw=3.6, ms=13,
                         label=METHOD_LABELS[method])

            lo, hi = 2e-3, 1.0
            ax.plot([lo, hi], [lo, hi], "--", color="#333333", lw=2.0, label="Perfect calibration")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, which="both", alpha=0.25)
            ax.tick_params(axis="both", labelsize=20)
            ax.set_title(f"{model} · {fs}", fontsize=25, fontweight="bold", pad=10)
            if i == 1:
                ax.set_xlabel("Mean predicted probability (log scale)", fontsize=22)
            if j == 0:
                ax.set_ylabel("Observed positive fraction (log scale)", fontsize=22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=24,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_png = OUT / "calibration_reliability_grid.png"
    out_pdf = OUT / "calibration_reliability_grid.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(out_png)
    print(out_pdf)


def plot_mapping_grid():
    # Sized/fonted for slide embedding — significantly larger than a print/PDF figure.
    fig, axes = plt.subplots(2, 3, figsize=(26.0, 20.0), gridspec_kw={"hspace": 0.45, "wspace": 0.3})

    for i, model in enumerate(MODELS):
        for j, fs in enumerate(FEATURE_SETS):
            ax = axes[i, j]
            slug = SLUGS[(model, fs)]
            raw_score, isotonic_score, platt_score = load_predictions(slug)

            order = np.argsort(raw_score)
            stride = max(1, len(order) // SUBSAMPLE_POINTS)
            sel = order[::stride]

            ax.plot(raw_score[sel], isotonic_score[sel], color=METHOD_COLORS["isotonic"],
                    lw=3.6, label=METHOD_LABELS["isotonic"])
            ax.plot(raw_score[sel], platt_score[sel], color=METHOD_COLORS["platt"],
                    lw=3.6, label=METHOD_LABELS["platt"])
            ax.plot([0, 1], [0, 1], "--", color="#333333", lw=2.0, label="y = x (no change)")

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=19)
            ax.set_title(f"{model} · {fs}", fontsize=24, fontweight="bold", pad=12)
            if i == 1:
                ax.set_xlabel("Raw score", fontsize=21)
            if j == 0:
                ax.set_ylabel("Calibrated score", fontsize=21)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=23,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Calibration Mapping", fontsize=40, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    out_png = OUT / "calibration_mapping_grid.png"
    out_pdf = OUT / "calibration_mapping_grid.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(out_png)
    print(out_pdf)


def main():
    plot_reliability_grid()
    plot_mapping_grid()


if __name__ == "__main__":
    main()
