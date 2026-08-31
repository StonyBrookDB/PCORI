"""
Confusion-matrix summary tables for the exp_calib_july pilot: Logistic
Regression and Transformer, each on Kelly / Rick / Full List 299, evaluated
on the (single, untouched) test split at real cohort prevalence.

Same row/column layout as the existing 10-fold-CV confusion table (see
plot_cv_real_prevalence_figure.py / 07_confusion_counts.py), but:
  - counts are exact integers from ONE held-out test split (not an average
    of 10 CV folds), since exp_calib_july is a single train/calibration/
    test split, not cross-validation
  - produced at TWO thresholds, each its own table:
      Threshold 0.5            — raw score >= 0.5 (the naive operating point)
      Threshold true prevalence — isotonic score >= isotonic's
                                  prevalence_matched threshold (same
                                  convention used for the ROC curve figures)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).parent.parent
EXP = ROOT / "experiments" / "exp_calib_july"
OUT = ROOT / "results" / "exp_calib_july" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    ("LR", "Kelly", "logistic_regression_kelly"),
    ("LR", "Rick", "logistic_regression_rick_top251"),
    ("LR", "Full List 299", "logistic_regression_top300"),
    ("Transformer", "Kelly", "transformer_kelly"),
    ("Transformer", "Rick", "transformer_rick_top251"),
    ("Transformer", "Full List 299", "transformer_top300"),
]


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def load_experiment(slug: str):
    z = np.load(EXP / slug / "predictions.npz")
    y_true = z["y_true"].astype(np.int8)
    raw_score = z["raw_score"].astype(np.float32)
    isotonic_score = z["isotonic_score"].astype(np.float32)
    summary = json.loads((EXP / slug / "summary_calib_split.json").read_text())
    return y_true, raw_score, isotonic_score, summary


def counts_at(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict:
    y_pred = (y_score >= thr).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    ppv = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    f1 = safe_div(2 * ppv * sens, ppv + sens)
    return {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv, "f1": f1}


def build_rows(threshold_kind: str) -> tuple[list[dict], int]:
    rows = []
    n_test = None
    for model, feature_set, slug in EXPERIMENTS:
        y_true, raw_score, isotonic_score, summary = load_experiment(slug)
        n_test = summary["split"]["n_test"]
        if threshold_kind == "default":
            c = counts_at(y_true, raw_score, 0.5)
        elif threshold_kind == "prevalence":
            thr = summary["methods"]["isotonic"]["prevalence_matched"]["threshold"]
            c = counts_at(y_true, isotonic_score, thr)
        else:
            raise ValueError(threshold_kind)
        rows.append({"model": f"{model}-{feature_set}", **c})
    return rows, n_test


def render_table(rows: list[dict], caption: str, footnote: str, out_stem: Path):
    col_labels = ["Model", "TP", "FN", "FP", "TN",
                  "Sensitivity/Recall", "Specificity", "PPV/Precision", "NPV", "F1"]
    cell_text = []
    for r in rows:
        cell_text.append([
            r["model"],
            f"{r['TP']:,}", f"{r['FN']:,}", f"{r['FP']:,}", f"{r['TN']:,}",
            f"{r['sensitivity']*100:.2f}%", f"{r['specificity']*100:.2f}%",
            f"{r['ppv']*100:.2f}%", f"{r['npv']*100:.2f}%", f"{r['f1']*100:.2f}%",
        ])

    n_rows = len(rows)
    fig_h = 1.1 + 0.52 * n_rows
    fig, ax = plt.subplots(figsize=(17.5, fig_h))
    ax.axis("off")

    ax.text(0.0, 1.0, caption, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=15, fontweight="normal")

    # column widths proportional to the longest string (header or cell) in
    # each column, with a little padding, normalized to sum to 1 — matplotlib's
    # ax.table() doesn't size columns to content on its own.
    col_chars = [
        max(len(col_labels[i]), max(len(row[i]) for row in cell_text)) + 2
        for i in range(len(col_labels))
    ]
    total_chars = sum(col_chars)
    col_widths = [c / total_chars for c in col_chars]

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="left",
                     colLoc="left", loc="center", bbox=[0.0, 0.14, 1.0, 0.80],
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(12.5)

    n_cols = len(col_labels)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("none")
        cell.PAD = 0.01
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(1.4)
            cell.set_edgecolor("#222222")
        elif row == 1:
            cell.visible_edges = ""
        if row == n_rows:
            cell.visible_edges = "B"
            cell.set_linewidth(1.0)
            cell.set_edgecolor("#222222")

    # double rule under the header, single rule above it and at the bottom
    for col in range(n_cols):
        header_cell = table[0, col]
        header_cell.visible_edges = "TB"

    ax.text(0.0, 0.02, footnote, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, color="#333333", wrap=True)

    fig.tight_layout()
    out_png = out_stem.with_suffix(".png")
    out_pdf = out_stem.with_suffix(".pdf")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(out_png)
    print(out_pdf)

    out_csv = out_stem.with_suffix(".csv")
    with out_csv.open("w", newline="") as f:
        fieldnames = ["model", "TP", "FN", "FP", "TN", "sensitivity", "specificity", "ppv", "npv", "f1"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(out_csv)


def main():
    default_rows, n_test = build_rows("default")
    render_table(
        default_rows,
        caption="Threshold 0.5:",
        footnote=(
            f"Values are exact confusion-matrix counts on the held-out test split "
            f"(exp_calib_july), evaluated at the real cohort prevalence. "
            f"The test split contains {n_test:,} patients (single train/calibration/"
            f"test split, not 10-fold cross-validation, so counts are not fold averages)."
        ),
        out_stem=OUT / "confusion_table_threshold_default_0p5",
    )

    prevalence_rows, n_test = build_rows("prevalence")
    render_table(
        prevalence_rows,
        caption="Threshold true prevalence:",
        footnote=(
            f"Values are exact confusion-matrix counts on the held-out test split "
            f"(exp_calib_july), evaluated at the real cohort prevalence. Threshold is "
            f"the isotonic-calibrated prevalence_matched cutoff, selected on the "
            f"calibration split only (never the test split). The test split contains "
            f"{n_test:,} patients (single train/calibration/test split, not 10-fold "
            f"cross-validation, so counts are not fold averages)."
        ),
        out_stem=OUT / "confusion_table_threshold_prevalence",
    )


if __name__ == "__main__":
    main()
