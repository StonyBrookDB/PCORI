"""
Compare threshold strategies: F1-optimal (swept) vs prevalence-matched
(clinician-calibrated) vs default 0.5.

Reads experiments/{slug}_{fs}/summary_calibration.json (written by
05_calibration_eval.py). Only models that have calibration results are plotted;
the rest are skipped with a note.

For each requested feature set it produces:
  results/calib_compare_f1_{fs}.png   grouped bars: F1 at default / f1_swept /
                                       prevalence_matched, per model
  results/calib_compare_pr_{fs}.png   Precision vs Recall at the two key
                                       operating points (shows that
                                       prevalence_matched gives P ≈ R)

Usage:
  python src/plot_calibration_compare.py                 # all feature sets
  python src/plot_calibration_compare.py --features top300
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model_registry import MODEL_REGISTRY

ROOT = Path(__file__).parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

FEATURE_SETS = ["top50", "top100", "top150", "top200", "top300"]
STRAT_LABELS = {
    "default": "Default (thr=0.5)",
    "f1_swept": "F1-optimal (swept)",
    "prevalence_matched": "Prevalence-matched",
}
STRAT_COLORS = {
    "default": "#bdbdbd",
    "f1_swept": "#1f77b4",
    "prevalence_matched": "#d62728",
}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_calib(model_name: str, fs: str):
    p = EXP / f"{slug(model_name)}_{fs}" / "summary_calibration.json"
    return json.loads(p.read_text()) if p.exists() else None


def is_fully_calibrated(model_name: str) -> bool:
    """True only if the model has calibration results for ALL feature sets.
    Partially-finished models are excluded from every plot."""
    return all((EXP / f"{slug(model_name)}_{fs}" / "summary_calibration.json").exists()
               for fs in FEATURE_SETS)


def plot_f1_strategies(fs: str):
    models, data = [], {}
    for entry in MODEL_REGISTRY:
        name = entry[0]
        if not is_fully_calibrated(name):
            continue
        s = load_calib(name, fs)
        if s is None:
            continue
        models.append(name)
        data[name] = s["strategies"]

    if not models:
        print(f"  [{fs}] no calibration results yet — skipping F1 plot")
        return

    strategies = ["default", "f1_swept", "prevalence_matched"]
    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(models)), 5.5))
    for i, strat in enumerate(strategies):
        means = [data[m][strat]["f1_mean"] for m in models]
        errs = [data[m][strat]["f1_std"] for m in models]
        ax.bar(x + (i - 1) * width, means, width, yerr=errs, capsize=3,
               label=STRAT_LABELS[strat], color=STRAT_COLORS[strat])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1 (mean ± std over 10 folds)")
    ax.set_title(f"F1 by threshold strategy — {fs}\n"
                 f"(optimal threshold vs prevalence-matched calibration)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUT / f"calib_compare_f1_{fs}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}  ({len(models)} models)")


def plot_pr_operating_points(fs: str):
    models, data = [], {}
    for entry in MODEL_REGISTRY:
        name = entry[0]
        if not is_fully_calibrated(name):
            continue
        s = load_calib(name, fs)
        if s is None:
            continue
        models.append(name)
        data[name] = s["strategies"]

    if not models:
        print(f"  [{fs}] no calibration results yet — skipping PR plot")
        return

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(models)), 5.5))
    # f1_swept P/R and prevalence_matched P/R
    series = [
        ("f1_swept", "precision", "F1-opt Precision", "#1f77b4"),
        ("f1_swept", "recall",    "F1-opt Recall",    "#aec7e8"),
        ("prevalence_matched", "precision", "PrevMatch Precision", "#d62728"),
        ("prevalence_matched", "recall",    "PrevMatch Recall",    "#ff9896"),
    ]
    for i, (strat, metric, label, color) in enumerate(series):
        vals = [data[m][strat][f"{metric}_mean"] for m in models]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("score")
    ax.set_title(f"Precision / Recall at each operating point — {fs}\n"
                 f"(prevalence-matched yields P ≈ R by construction)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUT / f"calib_compare_pr_{fs}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}  ({len(models)} models)")


def plot_threshold_difference(fs: str):
    """Two panels that make the optimal-vs-calibrated threshold difference obvious:
      left  — the threshold VALUES (dumbbell: F1-optimal vs prevalence-matched)
      right — the resulting operating point in Precision-Recall space (an arrow
              from the F1-optimal point to the prevalence-matched point; the
              dashed diagonal is P = R).
    """
    models, data = [], {}
    for entry in MODEL_REGISTRY:
        name = entry[0]
        if not is_fully_calibrated(name):
            continue
        s = load_calib(name, fs)
        if s is None:
            continue
        models.append(name)
        data[name] = s["strategies"]

    if not models:
        print(f"  [{fs}] no calibration results yet — skipping threshold-diff plot")
        return

    OPT, CAL = "f1_swept", "prevalence_matched"
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(models), 3)))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 0.6 * len(models) + 4))

    # ── Left: threshold value dumbbell ──
    y = np.arange(len(models))
    for i, m in enumerate(models):
        t_opt = data[m][OPT]["threshold_mean"]
        t_cal = data[m][CAL]["threshold_mean"]
        axL.plot([t_opt, t_cal], [i, i], color="#bbbbbb", lw=2, zorder=1)
        axL.scatter(t_opt, i, color="#1f77b4", s=90, zorder=2,
                    label="F1-optimal threshold" if i == 0 else None)
        axL.scatter(t_cal, i, color="#d62728", s=90, zorder=2,
                    label="Prevalence-matched threshold" if i == 0 else None)
        axL.annotate(f"Δ={t_cal - t_opt:+.3f}",
                     ((t_opt + t_cal) / 2, i + 0.18), ha="center", fontsize=8,
                     color="#555555")
    axL.set_yticks(y)
    axL.set_yticklabels(models, fontsize=9)
    axL.invert_yaxis()
    axL.set_xlabel("decision threshold")
    axL.set_title(f"Threshold value: F1-optimal vs prevalence-matched — {fs}")
    axL.legend(loc="lower left", fontsize=8)
    axL.grid(True, axis="x", alpha=0.3)

    # ── Right: operating point shift in P-R space ──
    vmin, vmax = 1.0, 0.0
    for i, m in enumerate(models):
        r_opt = data[m][OPT]["recall_mean"]; p_opt = data[m][OPT]["precision_mean"]
        r_cal = data[m][CAL]["recall_mean"]; p_cal = data[m][CAL]["precision_mean"]
        vmin = min(vmin, r_opt, p_opt, r_cal, p_cal)
        vmax = max(vmax, r_opt, p_opt, r_cal, p_cal)
        c = palette[i]
        axR.annotate("", xy=(r_cal, p_cal), xytext=(r_opt, p_opt),
                     arrowprops=dict(arrowstyle="->", color=c, lw=1.8))
        axR.scatter(r_opt, p_opt, color=c, marker="o", s=80, zorder=3)
        axR.scatter(r_cal, p_cal, color=c, marker="s", s=80, zorder=3, label=m)
    # tight, equal limits around the actual points (with a small margin)
    pad = max(0.02, (vmax - vmin) * 0.20)
    lo, hi = max(0.0, vmin - pad), min(1.0, vmax + pad)
    diag = np.linspace(lo, hi, 50)
    axR.plot(diag, diag, "--", color="#999999", lw=1, label="P = R")
    axR.set_xlim(lo, hi)
    axR.set_ylim(lo, hi)
    axR.set_aspect("equal", adjustable="box")
    axR.set_xlabel("Recall")
    axR.set_ylabel("Precision")
    axR.set_title(f"Operating point shift — {fs}\n"
                  f"○ F1-optimal  →  ■ prevalence-matched (lands on P=R)")
    axR.legend(fontsize=8, loc="upper right")
    axR.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT / f"calib_threshold_diff_{fs}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}  ({len(models)} models)")


def plot_combined_per_model(fs: str):
    """ONE figure that carries all four pieces of information per model:
        - optimal threshold value          (x-tick label of the left group)
        - calibrated threshold value       (x-tick label of the right group)
        - performance at optimal threshold (P/R/F1 bars, left group)
        - performance at calibrated thr    (P/R/F1 bars, right group)
    One small panel per model, arranged in a grid.
    """
    models, data = [], {}
    for entry in MODEL_REGISTRY:
        name = entry[0]
        if not is_fully_calibrated(name):
            continue
        s = load_calib(name, fs)
        if s is None:
            continue
        models.append(name)
        data[name] = s["strategies"]

    if not models:
        print(f"  [{fs}] no calibration results yet — skipping combined plot")
        return

    OPT, CAL = "f1_swept", "prevalence_matched"
    metrics = [("precision", "Precision", "#4c72b0"),
               ("recall", "Recall", "#dd8452"),
               ("f1", "F1", "#55a868")]

    ncols = min(len(models), 4)
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows),
                             squeeze=False)

    ymax = 0.0
    for m in models:
        for strat in (OPT, CAL):
            for k, _, _ in metrics:
                ymax = max(ymax, data[m][strat][f"{k}_mean"] + data[m][strat][f"{k}_std"])
    ymax = min(1.0, ymax * 1.18)

    handles = None
    for idx, m in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        groups = [OPT, CAL]
        gx = np.arange(2)              # 0 = model pick, 1 = doctor wants
        width = 0.25
        bars = []
        for j, (k, label, color) in enumerate(metrics):
            means = [data[m][g][f"{k}_mean"] for g in groups]
            errs = [data[m][g][f"{k}_std"] for g in groups]
            b = ax.bar(gx + (j - 1) * width, means, width, yerr=errs, capsize=3,
                       label=label, color=color)
            bars.append(b)
        if handles is None:
            handles = bars
        # annotate the THRESHOLD value above each operating-point group
        t_opt = data[m][OPT]["threshold_mean"]
        t_cal = data[m][CAL]["threshold_mean"]
        for xg, tv in zip(gx, [t_opt, t_cal]):
            ax.annotate(f"thr = {tv:.3f}", (xg, ymax * 0.99), ha="center", va="top",
                        fontsize=10, fontweight="bold", color="#333333")
        ax.set_xticks(gx)
        ax.set_xticklabels(["F1-optimal", "Prevalence-matched"], fontsize=9)
        ax.set_ylim(0, ymax)
        ax.set_title(m, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel("score")

    # hide any unused panels
    for j in range(len(models), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # single shared legend (avoids overlapping the threshold text)
    fig.legend([b[0] for b in handles], [lbl for _, lbl, _ in metrics],
               loc="upper right", fontsize=10, ncol=3, bbox_to_anchor=(0.99, 0.99))
    fig.suptitle(
        f"F1-optimal threshold vs Prevalence-matched threshold — {fs}\n"
        f"each panel: threshold value (bold) + Precision/Recall/F1 at that threshold",
        fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT / f"calib_combined_{fs}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}  ({len(models)} models)")


FEATURE_SIZES = {"top50": 50, "top100": 100, "top150": 150, "top200": 200, "top300": 299}


def plot_strategy_lines(metric: str = "f1"):
    """Line chart across feature-set sizes (top50→top300).
    One subplot per model; lines = F1 at default(0.5) / F1-optimal / prevalence-matched.
    Only feature sets that have calibration results are plotted.
    """
    OPT, CAL, DEF = "f1_swept", "prevalence_matched", "default"
    strat_style = [
        (DEF, "Default (0.5)",        "#bbbbbb", "o"),
        (OPT, "F1-optimal",           "#1f77b4", "s"),
        (CAL, "Prevalence-matched",   "#d62728", "^"),
    ]

    # only include models with calibration for ALL feature sets (full lines)
    models = []
    model_data = {}
    for entry in MODEL_REGISTRY:
        name = entry[0]
        if not is_fully_calibrated(name):
            continue
        models.append(name)
        model_data[name] = {fs: load_calib(name, fs)["strategies"] for fs in FEATURE_SETS}

    if not models:
        print("  no model has calibration for all feature sets yet — skipping line chart")
        return

    ncols = min(len(models), 4)
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.8 * nrows),
                             squeeze=False)

    for idx, m in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        runs = model_data[m]
        for strat, label, color, marker in strat_style:
            xs, ys, es = [], [], []
            for fs in FEATURE_SETS:
                if fs not in runs:
                    continue
                xs.append(FEATURE_SIZES[fs])
                ys.append(runs[fs][strat][f"{metric}_mean"])
                es.append(runs[fs][strat][f"{metric}_std"])
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=es, label=label, color=color, marker=marker,
                        markersize=6, linewidth=1.6, capsize=3)
        ax.set_title(m, fontsize=11, fontweight="bold")
        ax.set_xticks(list(FEATURE_SIZES.values()))
        ax.grid(True, alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel(f"{metric.upper()}")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Feature set size")
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    for j in range(len(models), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"{metric.upper()} by threshold strategy vs feature set size\n"
                 f"(F1-optimal vs Prevalence-matched across top50–top300)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT / f"calib_lines_{metric}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}  ({len(models)} models)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", default=FEATURE_SETS,
                        choices=FEATURE_SETS)
    args = parser.parse_args()

    # line charts span all feature sets — drawn once
    print("[lines across feature sets]")
    plot_strategy_lines("f1")

    for fs in args.features:
        print(f"[{fs}]")
        plot_combined_per_model(fs)
        plot_f1_strategies(fs)
        plot_pr_operating_points(fs)
        plot_threshold_difference(fs)


if __name__ == "__main__":
    main()
