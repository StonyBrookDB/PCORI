"""
Create two clinician-facing figures:
  1. Logistic Regression: Kelly vs Rick
  2. Transformer: Kelly vs Rick

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
    ("Transformer", "Kelly"): EXP / "transformer_kelly" / "fold_metrics.csv",
    ("Transformer", "Rick"): EXP / "transformer_rick_top251" / "fold_metrics.csv",
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


def load_counts(path: Path, pos_counts: list[int], neg_counts: list[int]) -> dict[str, float]:
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
    return {
        key: float(np.mean([r[key] for r in per_fold]))
        for key in ["TP", "FN", "FP", "TN", "Sensitivity", "Specificity", "PPV", "NPV"]
    }


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
        ax.text(
            cx + cell_w / 2,
            cy + cell_h * 0.60,
            f"{key} = {fmt_count(stats[key])}",
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            color=c,
        )
        ax.text(
            cx + cell_w / 2,
            cy + cell_h * 0.38,
            f"({subtitle})",
            ha="center",
            va="center",
            fontsize=10.5,
            color=c,
        )


def add_panel(ax, x: float, y: float, w: float, h: float, title: str, stats: dict[str, float], palette: dict[str, str]):
    ax.text(x + w / 2, y + h + 0.78, title, ha="center", va="bottom", fontsize=21, fontweight="bold")
    ax.text(
        x + w / 2,
        y + h + 0.47,
        f"Sensitivity {fmt_pct(stats['Sensitivity'])}   Specificity {fmt_pct(stats['Specificity'])}",
        ha="center",
        va="bottom",
        fontsize=11.5,
    )
    ax.text(
        x + w / 2,
        y + h + 0.22,
        f"PPV {fmt_pct(stats['PPV'])}   NPV {fmt_pct(stats['NPV'])}",
        ha="center",
        va="bottom",
        fontsize=11.5,
    )

    frame = FancyBboxPatch(
        (x - 0.35, y - 0.25),
        w + 0.7,
        h + 1.22,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        facecolor="none",
        edgecolor="#222222",
        lw=1.1,
    )
    ax.add_patch(frame)

    ax.text(x + w / 2, y + h + 0.02, "Predicted", ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.text(x + w * 0.25, y + h - 0.18, "Predicted\nHigh Risk", ha="center", va="top", fontsize=10.5)
    ax.text(x + w * 0.75, y + h - 0.18, "Predicted\nLow Risk", ha="center", va="top", fontsize=10.5)
    ax.text(x - 0.65, y + h * 0.75, "Actual\nHigh Risk", ha="center", va="center", fontsize=10.5)
    ax.text(x - 0.65, y + h * 0.25, "Actual\nLow Risk", ha="center", va="center", fontsize=10.5)
    ax.text(x - 1.05, y + h / 2, "Actual", ha="center", va="center", rotation=90, fontsize=15, fontweight="bold")
    add_matrix(ax, x, y, w, h, stats, palette)


def add_delta_arrow(ax, x1: float, x2: float, y: float, kelly: dict[str, float], rick: dict[str, float]):
    arrow = FancyArrowPatch(
        (x1, y),
        (x2, y),
        arrowstyle="Simple,tail_width=0.7,head_width=1.9,head_length=1.1",
        fc="#cdd8c9",
        ec="#333333",
        lw=0.9,
        mutation_scale=16,
    )
    ax.add_patch(arrow)
    dtp = rick["TP"] - kelly["TP"]
    dfp = rick["FP"] - kelly["FP"]
    dsens = rick["Sensitivity"] - kelly["Sensitivity"]
    dppv = rick["PPV"] - kelly["PPV"]
    ax.text(
        (x1 + x2) / 2,
        y + 0.05,
        f"+{dtp:,.1f} TP/fold\n+{dfp:,.1f} FP/fold\n"
        f"Delta Sens {dsens * 100:+.2f} pts | Delta PPV {dppv * 100:+.2f} pts",
        ha="center",
        va="center",
        fontsize=10.5,
    )


def add_summary(ax, x: float, y: float, w: float, model: str, kelly: dict[str, float], rick: dict[str, float]):
    dtp = rick["TP"] - kelly["TP"]
    dfn = rick["FN"] - kelly["FN"]
    dfp = rick["FP"] - kelly["FP"]
    box = FancyBboxPatch(
        (x, y),
        w,
        0.64,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor="#f8fbff",
        edgecolor="#2b2b2b",
        lw=1.0,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + 0.32,
        f"{model}: Rick identifies {dtp:,.1f} more true positives per fold "
        f"({-dfn:,.1f} fewer missed cases), with {dfp:,.1f} more false positives per fold.",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
    )


def plot_model(model: str, stats: dict[tuple[str, str], dict[str, float]], stem: str):
    blue = {"tp": "#173a67", "fn": "#c9d7e6", "fp": "#eef4fb", "tn": "#2e64aa"}
    green = {"tp": "#12682d", "fn": "#d4e8bb", "fp": "#72ad60", "tn": "#1f7436"}

    fig, ax = plt.subplots(figsize=(18, 7.2))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(9, 6.84, f"{model}: Kelly vs Rick", ha="center", va="center", fontsize=23, fontweight="bold")
    ax.text(
        9,
        6.47,
        "Real-prevalence confusion matrices, average per CV test fold",
        ha="center",
        va="center",
        fontsize=13,
    )
    ax.text(
        9,
        6.16,
        "10-fold cross-validation; 8 folds contain 1,179,186 patients and 2 folds contain 1,179,185 patients",
        ha="center",
        va="center",
        fontsize=11.5,
    )

    kelly = stats[(model, "Kelly")]
    rick = stats[(model, "Rick")]
    add_panel(ax, 2.7, 2.22, 4.9, 2.75, "Kelly", kelly, blue)
    add_panel(ax, 11.1, 2.22, 4.9, 2.75, "Rick", rick, green)
    add_delta_arrow(ax, 8.0, 10.4, 5.40, kelly, rick)
    add_summary(ax, 4.1, 1.29, 9.8, model, kelly, rick)

    ax.text(
        9,
        0.25,
        "Counts are means per test fold and retain the original cohort prevalence; they are not pooled totals across all folds.",
        ha="center",
        va="center",
        fontsize=11,
    )

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
    plot_model("Logistic Regression", stats, "cv_real_prevalence_lr_kelly_vs_rick")
    plot_model("Transformer", stats, "cv_real_prevalence_transformer_kelly_vs_rick")


if __name__ == "__main__":
    main()
