"""
Create two clinician-facing figures:
  1. Logistic Regression: Kelly vs Rick vs Full List 299
  2. Transformer: Kelly vs Rick vs Full List 299

Counts are average per-test-fold confusion-matrix counts from 10-fold
cross-validation, evaluated at the original cohort prevalence.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
EXP = ROOT / "experiments"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

N_SPLITS = 10
RANDOM_STATE = 42

EXPERIMENTS = {
    ("Logistic Regression", "Kelly"): EXP / "logistic_regression_kelly" / "fold_metrics.csv",
    ("Logistic Regression", "Rick"): EXP / "logistic_regression_rick_top251" / "fold_metrics.csv",
    ("Logistic Regression", "Full List 299"): EXP / "logistic_regression_top300" / "fold_predictions.npz",
    ("Transformer", "Kelly"): EXP / "transformer_kelly" / "fold_metrics.csv",
    ("Transformer", "Rick"): EXP / "transformer_rick_top251" / "fold_metrics.csv",
    ("Transformer", "Full List 299"): EXP / "transformer_top300" / "fold_predictions.npz",
}


def fold_sizes() -> tuple[list[int], list[int]]:
    y = np.load(DATA / "labels.npy")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pos_counts: list[int] = []
    neg_counts: list[int] = []
    for _, test_idx in skf.split(y, y):
        yy = y[test_idx]
        pos_counts.append(int(yy.sum()))
        neg_counts.append(int((yy == 0).sum()))
    return pos_counts, neg_counts


def summarize(per_fold: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in per_fold]))
        for key in ["TP", "FN", "FP", "TN", "Sensitivity", "Specificity", "PPV", "NPV"]
    }


def load_from_metrics(path: Path, pos_counts: list[int], neg_counts: list[int]) -> dict[str, float]:
    rows = list(csv.DictReader(path.open()))
    per_fold = []
    for i, row in enumerate(rows):
        p = pos_counts[i]
        n = neg_counts[i]
        tp = round(float(row["sensitivity"]) * p)
        tn = round(float(row["specificity"]) * n)
        per_fold.append(
            {
                "TP": tp,
                "FN": p - tp,
                "FP": n - tn,
                "TN": tn,
                "Sensitivity": float(row["sensitivity"]),
                "Specificity": float(row["specificity"]),
                "PPV": float(row["ppv"]),
                "NPV": float(row["npv"]),
            }
        )
    return summarize(per_fold)


def load_from_predictions(path: Path) -> dict[str, float]:
    z = np.load(path)
    folds = z["fold"]
    y_true = z["y_true"].astype(np.int8)
    y_score = z["y_score"].astype(np.float32)
    per_fold = []
    for fold in sorted(np.unique(folds)):
        sel = folds == fold
        yt = y_true[sel]
        yp = (y_score[sel] >= 0.5).astype(np.int8)
        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        per_fold.append(
            {
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "Sensitivity": tp / max(tp + fn, 1),
                "Specificity": tn / max(tn + fp, 1),
                "PPV": tp / max(tp + fp, 1),
                "NPV": tn / max(tn + fn, 1),
            }
        )
    return summarize(per_fold)


def load_counts(path: Path, pos_counts: list[int], neg_counts: list[int]) -> dict[str, float]:
    if path.suffix == ".npz":
        return load_from_predictions(path)
    return load_from_metrics(path, pos_counts, neg_counts)


def fmt_count(x: float) -> str:
    return f"{x:,.1f}"


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def text_color(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 115 else "black"


def add_matrix(ax, x: float, y: float, w: float, h: float, stats: dict[str, float], palette: dict[str, str]):
    cell_w = w / 2
    cell_h = h / 2
    cells = [
        ("TP", "Correct positives", x, y + cell_h, palette["tp"]),
        ("FN", "Missed cases", x + cell_w, y + cell_h, palette["fn"]),
        ("FP", "False positives", x, y, palette["fp"]),
        ("TN", "Correct negatives", x + cell_w, y, palette["tn"]),
    ]
    for key, subtitle, cx, cy, color in cells:
        ax.add_patch(Rectangle((cx, cy), cell_w, cell_h, facecolor=color, edgecolor="#222222", lw=1.2))
        c = text_color(color)
        ax.text(cx + cell_w / 2, cy + cell_h * 0.60, f"{key} = {fmt_count(stats[key])}",
                ha="center", va="center", fontsize=15.5, fontweight="bold", color=c)
        ax.text(cx + cell_w / 2, cy + cell_h * 0.38, f"({subtitle})",
                ha="center", va="center", fontsize=9.5, color=c)


def add_panel(ax, x: float, y: float, w: float, h: float, title: str, stats: dict[str, float], palette: dict[str, str]):
    ax.text(x + w / 2, y + h + 0.80, title, ha="center", va="bottom", fontsize=18, fontweight="bold")
    ax.text(x + w / 2, y + h + 0.50,
            f"Sensitivity {fmt_pct(stats['Sensitivity'])}   Specificity {fmt_pct(stats['Specificity'])}",
            ha="center", va="bottom", fontsize=10.2)
    ax.text(x + w / 2, y + h + 0.25, f"PPV {fmt_pct(stats['PPV'])}   NPV {fmt_pct(stats['NPV'])}",
            ha="center", va="bottom", fontsize=10.2)

    frame = FancyBboxPatch(
        (x - 0.32, y - 0.25),
        w + 0.64,
        h + 1.22,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        facecolor="none",
        edgecolor="#222222",
        lw=1.1,
    )
    ax.add_patch(frame)
    ax.text(x + w / 2, y + h + 0.02, "Predicted", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.text(x + w * 0.25, y + h - 0.16, "Predicted\nHigh Risk", ha="center", va="top", fontsize=9.6)
    ax.text(x + w * 0.75, y + h - 0.16, "Predicted\nLow Risk", ha="center", va="top", fontsize=9.6)
    ax.text(x - 0.58, y + h * 0.75, "Actual\nHigh Risk", ha="center", va="center", fontsize=9.6)
    ax.text(x - 0.58, y + h * 0.25, "Actual\nLow Risk", ha="center", va="center", fontsize=9.6)
    ax.text(x - 0.94, y + h / 2, "Actual", ha="center", va="center", rotation=90, fontsize=14, fontweight="bold")
    add_matrix(ax, x, y, w, h, stats, palette)


def add_delta_arrow(ax, x1: float, x2: float, y: float, start: dict[str, float], end: dict[str, float]):
    arrow = FancyArrowPatch(
        (x1, y),
        (x2, y),
        arrowstyle="Simple,tail_width=0.62,head_width=1.75,head_length=0.95",
        fc="#cdd8c9",
        ec="#333333",
        lw=0.9,
        mutation_scale=16,
    )
    ax.add_patch(arrow)
    dtp = end["TP"] - start["TP"]
    dfp = end["FP"] - start["FP"]
    dsens = end["Sensitivity"] - start["Sensitivity"]
    dppv = end["PPV"] - start["PPV"]
    ax.text((x1 + x2) / 2, y + 0.04,
            f"{dtp:+,.1f} TP/fold\n{dfp:+,.1f} FP/fold\n"
            f"Delta Sens {dsens * 100:+.2f} pts | Delta PPV {dppv * 100:+.2f} pts",
            ha="center", va="center", fontsize=8.8)


def add_summary(
    ax,
    x: float,
    y: float,
    w: float,
    model: str,
    kelly: dict[str, float],
    rick: dict[str, float],
    full: dict[str, float],
):
    rick_dtp = rick["TP"] - kelly["TP"]
    rick_dfp = rick["FP"] - kelly["FP"]
    full_dtp = full["TP"] - kelly["TP"]
    full_dfp = full["FP"] - kelly["FP"]
    box = FancyBboxPatch(
        (x, y),
        w,
        0.70,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor="#f8fbff",
        edgecolor="#2b2b2b",
        lw=1.0,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + 0.35,
        f"{model}: versus Kelly, Rick adds {rick_dtp:,.1f} TP/fold and {rick_dfp:,.1f} FP/fold; "
        f"Full List 299 adds {full_dtp:,.1f} TP/fold and {full_dfp:,.1f} FP/fold.",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )


def plot_model(model: str, stats: dict[tuple[str, str], dict[str, float]], stem: str):
    blue = {"tp": "#173a67", "fn": "#c9d7e6", "fp": "#eef4fb", "tn": "#2e64aa"}
    green = {"tp": "#12682d", "fn": "#d4e8bb", "fp": "#72ad60", "tn": "#1f7436"}
    gray = {"tp": "#374151", "fn": "#d7dde5", "fp": "#f0f3f7", "tn": "#667085"}

    fig, ax = plt.subplots(figsize=(22, 7.5))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 7.25)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(11, 6.88, f"{model}: Kelly vs Rick vs Full List 299",
            ha="center", va="center", fontsize=23, fontweight="bold")
    ax.text(11, 6.50, "Real-prevalence confusion matrices, average per CV test fold",
            ha="center", va="center", fontsize=13)
    ax.text(11, 6.18,
            "10-fold cross-validation; 8 folds contain 1,179,186 patients and 2 folds contain 1,179,185 patients",
            ha="center", va="center", fontsize=11.5)

    kelly = stats[(model, "Kelly")]
    rick = stats[(model, "Rick")]
    full = stats[(model, "Full List 299")]
    y = 2.25
    add_panel(ax, 1.55, y, 4.25, 2.72, "Kelly", kelly, blue)
    add_panel(ax, 8.70, y, 4.25, 2.72, "Rick", rick, green)
    add_panel(ax, 15.85, y, 4.25, 2.72, "Full List 299\nwithout human curation", full, gray)
    add_delta_arrow(ax, 6.20, 8.10, 5.42, kelly, rick)
    add_delta_arrow(ax, 13.35, 15.25, 5.42, rick, full)
    add_summary(ax, 3.85, 1.25, 14.30, model, kelly, rick, full)

    ax.text(11, 0.25,
            "Counts are means per test fold and retain the original cohort prevalence; they are not pooled totals across all folds.",
            ha="center", va="center", fontsize=11)

    out_png = OUT / f"{stem}.png"
    out_pdf = OUT / f"{stem}.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(out_png)
    print(out_pdf)


def main():
    pos_counts, neg_counts = fold_sizes()
    stats = {
        key: load_counts(path, pos_counts, neg_counts)
        for key, path in EXPERIMENTS.items()
    }
    plot_model("Logistic Regression", stats, "cv_real_prevalence_lr_kelly_rick_full299")
    plot_model("Transformer", stats, "cv_real_prevalence_transformer_kelly_rick_full299")


if __name__ == "__main__":
    main()
