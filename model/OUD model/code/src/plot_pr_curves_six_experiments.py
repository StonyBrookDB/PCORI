"""
Plot real-prevalence precision-recall curves for the Kelly/Rick/Full List 299
experiments for Logistic Regression and Transformer.

Curves require per-patient out-of-fold scores in fold_predictions.npz. If an
experiment only has fold-level metrics, it is reported as missing and skipped;
the script does not fabricate a curve from aggregate metrics.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

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


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    z = np.load(path)
    return z["y_true"].astype(np.int8), z["y_score"].astype(np.float32)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.1), sharey=True)
    rows = []
    missing = []

    for model, feature_set, path in EXPERIMENTS:
        ax = axes[0] if model == "Logistic Regression" else axes[1]
        data = load_predictions(path)
        if data is None:
            missing.append((model, feature_set, str(path)))
            rows.append(
                {
                    "model": model,
                    "feature_set": feature_set,
                    "n": "",
                    "prevalence": "",
                    "average_precision": "",
                    "status": "missing fold_predictions.npz",
                }
            )
            continue

        y_true, y_score = data
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        prevalence = float(y_true.mean())
        label = f"{feature_set} (AUPRC={ap:.3f})"
        ax.plot(
            recall,
            precision,
            color=COLORS.get(feature_set),
            lw=1.9,
            label=label,
        )
        rows.append(
            {
                "model": model,
                "feature_set": feature_set,
                "n": len(y_true),
                "prevalence": f"{prevalence:.6f}",
                "average_precision": f"{ap:.6f}",
                "status": "plotted",
            }
        )

    for ax, title in zip(axes, ["Logistic Regression", "Transformer"]):
        ax.axhline(0.011637, color="#999999", ls="--", lw=1.0, label="Cohort prevalence=1.16%")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.32)
        ax.set_xlabel("Recall / Sensitivity")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right", fontsize=9, frameon=True)
    axes[0].set_ylabel("Precision / PPV")

    if missing:
        missing_text = "; ".join([f"{m} {fs}" for m, fs, _ in missing])
        fig.text(
            0.5,
            0.02,
            f"Skipped because per-patient scores are unavailable: {missing_text}. "
            "No PR curve was imputed from aggregate metrics.",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(
        "Precision-Recall Curves, Real Prevalence, Pooled Out-of-Fold Predictions",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    out_png = OUT / "pr_curves_lr_transformer_kelly_rick_full299.png"
    out_pdf = OUT / "pr_curves_lr_transformer_kelly_rick_full299.pdf"
    out_csv = OUT / "pr_curves_lr_transformer_kelly_rick_full299_summary.csv"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "feature_set", "n", "prevalence", "average_precision", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(out_png)
    print(out_pdf)
    print(out_csv)
    if missing:
        print("Missing curves:")
        for model, feature_set, path in missing:
            print(f"  - {model} {feature_set}: {path}")


if __name__ == "__main__":
    main()
