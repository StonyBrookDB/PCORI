"""
Plot ROC curves for the six Kelly/Rick/Full List 299 experiments:
Logistic Regression and Transformer, each with three feature sets.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

EXPERIMENTS = [
    ("Logistic Regression", "Kelly", EXP / "logistic_regression_kelly" / "fold_predictions.npz"),
    ("Logistic Regression", "Rick", EXP / "logistic_regression_rick_top251" / "fold_predictions.npz"),
    ("Logistic Regression", "Full List 299", EXP / "logistic_regression_top300" / "fold_predictions.npz"),
    ("Transformer", "Kelly", EXP / "transformer_kelly" / "fold_predictions.npz"),
    ("Transformer", "Rick", EXP / "transformer_rick_top251" / "fold_predictions.npz"),
    ("Transformer", "Full List 299", EXP / "transformer_top300" / "fold_predictions.npz"),
]

COLORS = {
    "Kelly": "#1f77b4",
    "Rick": "#2ca02c",
    "Full List 299": "#4b5563",
}


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["y_true"].astype(np.int8), z["y_score"].astype(np.float32)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.1), sharey=True)
    rows = []

    for model, feature_set, path in EXPERIMENTS:
        if not path.exists():
            raise FileNotFoundError(path)
        y_true, y_score = load_predictions(path)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax = axes[0] if model == "Logistic Regression" else axes[1]
        ax.plot(
            fpr,
            tpr,
            color=COLORS[feature_set],
            lw=1.9,
            label=f"{feature_set} (AUROC={auc:.3f})",
        )
        rows.append(
            {
                "model": model,
                "feature_set": feature_set,
                "n": len(y_true),
                "prevalence": f"{float(y_true.mean()):.6f}",
                "auroc": f"{auc:.6f}",
            }
        )

    for ax, title in zip(axes, ["Logistic Regression", "Transformer"]):
        ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1.0, label="Random")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="lower right", fontsize=9, frameon=True)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")

    fig.suptitle(
        "ROC Curves, Real Prevalence, Pooled Out-of-Fold Predictions",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_png = OUT / "roc_curves_lr_transformer_kelly_rick_full299.png"
    out_pdf = OUT / "roc_curves_lr_transformer_kelly_rick_full299.pdf"
    out_csv = OUT / "roc_curves_lr_transformer_kelly_rick_full299_summary.csv"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "feature_set", "n", "prevalence", "auroc"])
        writer.writeheader()
        writer.writerows(rows)

    print(out_png)
    print(out_pdf)
    print(out_csv)


if __name__ == "__main__":
    main()
