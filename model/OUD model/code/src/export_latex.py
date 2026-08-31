"""
Export paper-ready LaTeX three-line (booktabs) tables and figure blocks
from the existing experiment outputs.

Reads:
  experiments/{slug}_{fs}/summary.json
  experiments/{slug}_{fs}/summary_calibration.json
  experiments/{slug}_{fs}/fold_metrics.csv      (for paired t-test p-values)

Writes:
  results/tables.tex          — all three-line tables in one file (\\input-able)
  results/figures.tex         — \\includegraphics blocks for the key PNGs
  results/paper_bundle.tex    — minimal standalone wrapper that includes both

Usage:
  python src/export_latex.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).parent))
from model_registry import MODEL_REGISTRY

ROOT = Path(__file__).parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

FEATURE_SETS = ["top50", "top100", "top150", "top200", "top300"]
FEATURE_SIZES = {"top50": 50, "top100": 100, "top150": 150, "top200": 200, "top300": 299}
REFERENCE_MODEL = "Random Forest"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt(mean, std=None, prec=4):
    if mean is None:
        return "--"
    if std is None:
        return f"{mean:.{prec}f}"
    return f"${mean:.{prec}f} \\pm {std:.{prec}f}$"


def pvalue_tex(p):
    if p is None:
        return "--"
    if p < 0.001:
        return r"$<\!0.001$"
    if p < 0.01:
        return r"$<\!0.01$"
    if p < 0.05:
        return r"$<\!0.05$"
    return f"{p:.3f}"


def load_main(model, fs):
    p = EXP / f"{slug(model)}_{fs}" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_calib(model, fs):
    p = EXP / f"{slug(model)}_{fs}" / "summary_calibration.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_extended(model, fs):
    """summary_extended.json — Sens / Spec / PPV / NPV / AUPRC + balanced_*
    metrics (backfilled by src/06_backfill_extended_metrics.py for old runs,
    written directly by src/03_train_eval.py for new runs)."""
    p = EXP / f"{slug(model)}_{fs}" / "summary_extended.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_full(model, fs):
    """Merge main + extended; extended overrides where keys collide."""
    out = {}
    main = load_main(model, fs)
    ext = load_extended(model, fs)
    if main: out.update(main)
    if ext:  out.update(ext)
    return out if out else None


def load_fold_vals(model, fs, columns):
    """Return dict {col: np.array of 10 fold values}."""
    p = EXP / f"{slug(model)}_{fs}" / "fold_metrics.csv"
    if not p.exists():
        return None
    out = {c: [] for c in columns}
    with open(p) as f:
        for row in csv.DictReader(f):
            for c in columns:
                if row.get(c):
                    out[c].append(float(row[c]))
    return {c: np.array(v) for c, v in out.items()}


# ────────────────────────────────────────────────────────────────────────────
# Table 1: Main results at a given feature set (default top300)
# ────────────────────────────────────────────────────────────────────────────
def table_main_at_fs(fs: str = "top300") -> str:
    rows = []
    ref_vals = None
    # collect fold values for p-tests
    fold_cache = {}
    for entry in MODEL_REGISTRY:
        name = entry[0]
        fold_cache[name] = load_fold_vals(name, fs, ["f1", "auroc", "f1_best"]) or {}
        if name == REFERENCE_MODEL:
            ref_vals = fold_cache[name]

    for entry in MODEL_REGISTRY:
        name = entry[0]
        s = load_main(name, fs)
        if s is None:
            continue
        if name == REFERENCE_MODEL or ref_vals is None:
            p_f1 = p_au = None
        else:
            cur = fold_cache[name]
            p_f1 = ttest_rel(cur["f1"], ref_vals["f1"])[1] if len(cur.get("f1", [])) and len(ref_vals.get("f1", [])) else None
            p_au = ttest_rel(cur["auroc"], ref_vals["auroc"])[1] if len(cur.get("auroc", [])) and len(ref_vals.get("auroc", [])) else None
        rows.append({
            "model": name,
            "P":  (s["precision_mean"], s["precision_std"]),
            "R":  (s["recall_mean"], s["recall_std"]),
            "F1": (s["f1_mean"], s["f1_std"]),
            "AU": (s["auroc_mean"], s["auroc_std"]),
            "F1s": (s.get("f1_best_mean"), s.get("f1_best_std")),
            "p_f1": p_f1, "p_au": p_au,
        })

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Main results on {fs} ({FEATURE_SIZES[fs]} features). "
                 r"All metrics are reported as mean $\pm$ standard deviation over 10-fold "
                 r"stratified cross-validation. Precision, Recall, and F1 use the default "
                 r"threshold of 0.5; F1$^{*}$ is F1 at the F1-optimal threshold per fold. "
                 r"P-values are from paired $t$-tests against the Random Forest reference.}")
    lines.append(rf"\label{{tab:main_{fs}}}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Precision & Recall & F1 & AUROC & F1$^{*}$ & $p$(F1) & $p$(AUROC) \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(
            f"{latex_escape(r['model'])} & "
            f"{fmt(*r['P'])} & {fmt(*r['R'])} & {fmt(*r['F1'])} & "
            f"{fmt(*r['AU'])} & {fmt(*r['F1s'])} & "
            f"{pvalue_tex(r['p_f1'])} & {pvalue_tex(r['p_au'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Table 2: AUROC (or F1*) across all 5 feature sets — scaling table
# ────────────────────────────────────────────────────────────────────────────
def table_scaling(metric_key: str, metric_label: str, caption: str) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{tab:scaling_{metric_key}}}")
    lines.append(r"\begin{tabular}{l" + "c" * len(FEATURE_SETS) + r"}")
    lines.append(r"\toprule")
    head = " & ".join([f"top-{FEATURE_SIZES[fs]}" for fs in FEATURE_SETS])
    lines.append(rf"Model & {head} \\")
    lines.append(r"\midrule")

    # find best per column across models
    best_per_col = {}
    for fs in FEATURE_SETS:
        best_val = -1.0; best_model = None
        for entry in MODEL_REGISTRY:
            name = entry[0]
            s = load_main(name, fs)
            if s is None:
                continue
            v = s.get(f"{metric_key}_mean")
            if v is not None and v > best_val:
                best_val, best_model = v, name
        best_per_col[fs] = best_model

    for entry in MODEL_REGISTRY:
        name = entry[0]
        cells = []
        any_present = False
        for fs in FEATURE_SETS:
            s = load_main(name, fs)
            if s is None:
                cells.append("--")
                continue
            any_present = True
            v = s.get(f"{metric_key}_mean")
            sd = s.get(f"{metric_key}_std")
            text = fmt(v, sd)
            if best_per_col[fs] == name:
                # bold math properly: $...$ -> $\mathbf{...}$
                inner = text.strip("$")
                text = r"$\mathbf{" + inner + r"}$"
            cells.append(text)
        if not any_present:
            continue
        lines.append(f"{latex_escape(name)} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Table 3: Threshold calibration — F1-optimal vs Prevalence-matched at top300
# ────────────────────────────────────────────────────────────────────────────
def table_calibration(fs: str = "top300") -> str:
    rows = []
    for entry in MODEL_REGISTRY:
        name = entry[0]
        cal = load_calib(name, fs)
        if cal is None or "strategies" not in cal:
            continue
        opt = cal["strategies"]["f1_swept"]
        pre = cal["strategies"]["prevalence_matched"]
        rows.append({
            "model": name,
            "auroc": (cal["auroc_mean"], cal["auroc_std"]),
            "t_opt": opt["threshold_mean"],
            "p_opt": (opt["precision_mean"], opt["precision_std"]),
            "r_opt": (opt["recall_mean"], opt["recall_std"]),
            "f_opt": (opt["f1_mean"], opt["f1_std"]),
            "t_cal": pre["threshold_mean"],
            "p_cal": (pre["precision_mean"], pre["precision_std"]),
            "r_cal": (pre["recall_mean"], pre["recall_std"]),
            "f_cal": (pre["f1_mean"], pre["f1_std"]),
        })

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Threshold calibration on {fs} ({FEATURE_SIZES[fs]} features): "
                 r"F1-optimal threshold vs prevalence-matched (clinical) threshold. "
                 r"Mean $\pm$ standard deviation over 10 folds. The prevalence-matched "
                 r"threshold yields Precision $\approx$ Recall by construction.}")
    lines.append(rf"\label{{tab:calibration_{fs}}}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{lccccccccc}")
    lines.append(r"\toprule")
    lines.append(r" &  & \multicolumn{4}{c}{F1-optimal threshold}"
                 r" & \multicolumn{4}{c}{Prevalence-matched threshold} \\")
    lines.append(r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}")
    lines.append(r"Model & AUROC & Threshold & Precision & Recall & F1"
                 r" & Threshold & Precision & Recall & F1 \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(
            f"{latex_escape(r['model'])} & {fmt(*r['auroc'])} & "
            f"{fmt(r['t_opt'], prec=3)} & {fmt(*r['p_opt'])} & {fmt(*r['r_opt'])} & {fmt(*r['f_opt'])} & "
            f"{fmt(r['t_cal'], prec=3)} & {fmt(*r['p_cal'])} & {fmt(*r['r_cal'])} & {fmt(*r['f_cal'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# figures.tex
# ────────────────────────────────────────────────────────────────────────────
def build_figures_tex() -> str:
    figs = [
        ("calib_lines_f1.png",
         r"F1 by threshold strategy across feature set sizes (top-50 to top-300). "
         r"For every model, F1 at the F1-optimal threshold (blue) and at the "
         r"prevalence-matched threshold (red) are nearly identical, while F1 at "
         r"the default 0.5 threshold (grey) is much lower.",
         "fig:calib_lines"),
        ("calib_combined_top300.png",
         r"Operating-point comparison on top-300 features. Each panel shows one "
         r"model's threshold value (bold) and Precision/Recall/F1 at that "
         r"threshold for both the F1-optimal and prevalence-matched operating "
         r"points. The prevalence-matched threshold yields balanced Precision and "
         r"Recall (the three bars become equal height).",
         "fig:calib_combined"),
        ("calib_threshold_diff_top300.png",
         r"Threshold values and their effect on the operating point. Left: each "
         r"model's F1-optimal threshold (blue) and prevalence-matched threshold "
         r"(red). Right: the resulting shift in Precision–Recall space; arrows go "
         r"from the F1-optimal point (circle) to the prevalence-matched point "
         r"(square), which lands on the $P{=}R$ diagonal.",
         "fig:threshold_diff"),
    ]
    out = []
    for fname, caption, label in figs:
        out.append(r"\begin{figure}[t]")
        out.append(r"\centering")
        out.append(rf"\includegraphics[width=\linewidth]{{results/{fname}}}")
        out.append(rf"\caption{{{caption}}}")
        out.append(rf"\label{{{label}}}")
        out.append(r"\end{figure}")
        out.append("")
    return "\n".join(out)


# ────────────────────────────────────────────────────────────────────────────
# entrypoint
# ────────────────────────────────────────────────────────────────────────────
PAPER_SECTIONS = [
    ("Traditional Methods (on raw feature input matrix)",
     ["Random Forest", "Decision Tree", "Logistic Regression", "DNN"]),
    ("Sequential Models (on raw feature input matrix)",
     ["LSTM", "Bi-LSTM", "Attention", "Transformer"]),
    ("Graph Models (on graph embeddings)",
     ["GCN", "GAT", "HeteroRGCN"]),
    ("Sequential Graph Combined Models (on both raw input and graph embeddings)",
     ["LSTM-GNN", "LSTM-GCN", "LSTM-GAT"]),
    ("Proposed Model",
     ["LIGHTED (LSTM-HeteroRGNN)"]),
]
PROPOSED_MODEL = "LIGHTED (LSTM-HeteroRGNN)"


def table_paper_style_grouped(fs: str = "top300") -> str:
    """Paper-style Table 2: grouped sections, bold = best within section per
    metric, '--' for not-yet-run models, '*' marker on the proposed model's
    p-value column. If the proposed model is not yet present, p-values are
    computed against Random Forest as a placeholder reference (noted in caption).
    """
    # ── load everything once ──
    main_by = {}        # model -> summary.json | None
    fold_by = {}        # model -> {f1: [...], auroc: [...]} | {}
    for _, models in PAPER_SECTIONS:
        for m in models:
            main_by[m] = load_main(m, fs)
            fold_by[m] = load_fold_vals(m, fs, ["f1", "auroc"]) or {}

    # reference: prefer proposed model if present, else Random Forest
    if main_by.get(PROPOSED_MODEL) is not None:
        reference = PROPOSED_MODEL
        ref_note = ""
    else:
        reference = "Random Forest"
        ref_note = (" Reference model for paired $t$-tests is Random Forest "
                    "(placeholder until the proposed model is available; "
                    "the proposed-model row is marked with `--' for now).")
    ref_vals = fold_by.get(reference, {})

    # ── best-per-section-per-metric ──
    METRICS = [("precision_mean", "precision_std"),
               ("recall_mean", "recall_std"),
               ("f1_mean", "f1_std"),
               ("auroc_mean", "auroc_std")]
    bests = {}
    for sec_name, models in PAPER_SECTIONS:
        for mean_key, _ in METRICS:
            best_v, best_m = -1.0, None
            for m in models:
                s = main_by.get(m)
                if s and s.get(mean_key) is not None and s[mean_key] > best_v:
                    best_v, best_m = s[mean_key], m
            bests[(sec_name, mean_key)] = best_m

    # ── render ──
    def bold_math(text: str) -> str:
        return r"$\mathbf{" + text.strip("$") + r"}$"

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Summary of prediction performance on {fs} ({FEATURE_SIZES[fs]} "
        r"features). Results are mean $\pm$ standard deviation over 10-fold "
        r"stratified cross-validation. The best result per metric within each "
        r"section is in \textbf{bold}; `--' marks models that have not yet been "
        r"trained." + ref_note + "}")
    lines.append(rf"\label{{tab:paper_table2_{fs}}}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Precision & Recall & F-1 & AUROC & $p$-value (F-1) & $p$-value (AUROC) \\")
    lines.append(r"\midrule")

    for sec_i, (sec_name, models) in enumerate(PAPER_SECTIONS):
        lines.append(
            r"\multicolumn{7}{l}{\textit{" + latex_escape(sec_name) + r"}} \\")
        for m in models:
            s = main_by.get(m)
            if s is None:
                cells = ["--"] * 6
            else:
                # P, R, F1, AUROC with bold if best in section
                metric_cells = []
                for mean_key, std_key in METRICS:
                    text = fmt(s[mean_key], s[std_key])
                    if bests[(sec_name, mean_key)] == m:
                        text = bold_math(text)
                    metric_cells.append(text)
                # p-values
                if m == reference:
                    p_f1 = p_au = r"$*$"
                else:
                    cur = fold_by.get(m, {})
                    if (cur.get("f1") is not None and len(cur.get("f1", [])) > 1
                            and len(ref_vals.get("f1", [])) > 1):
                        p_f1 = pvalue_tex(ttest_rel(cur["f1"], ref_vals["f1"])[1])
                        p_au = pvalue_tex(ttest_rel(cur["auroc"], ref_vals["auroc"])[1])
                    else:
                        p_f1 = p_au = "--"
                cells = metric_cells + [p_f1, p_au]
            lines.append(f"{latex_escape(m)} & " + " & ".join(cells) + r" \\")
        if sec_i < len(PAPER_SECTIONS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Paper-revision tables (LR + Transformer only, clinical metric set)
# ────────────────────────────────────────────────────────────────────────────
CLINICAL_MODELS = ["Logistic Regression", "Transformer"]
RICK_SIZES = ["rick_top50", "rick_top100", "rick_top150", "rick_top200", "rick_top251"]
RICK_SIZE_LABELS = {"rick_top50": 50, "rick_top100": 100, "rick_top150": 150,
                    "rick_top200": 200, "rick_top251": 251}

# Display order chosen to match the slide:
#   Sensitivity, Specificity, PPV, NPV, AUROC, F1, AUPRC
# All metrics are read from the BALANCED-test block (paper-comparable numbers).
CLINICAL_METRICS = [
    ("Sens.",  "balanced_recall"),
    ("Spec.",  "balanced_specificity"),
    ("PPV",    "balanced_precision"),
    ("NPV",    "balanced_npv"),
    ("AUROC",  "balanced_auroc"),
    ("F1",     "balanced_f1"),
    ("AUPRC",  "balanced_auprc"),
]


def _clinical_row_means(s: dict | None) -> list[float | None]:
    if s is None:
        return [None] * len(CLINICAL_METRICS)
    return [s.get(f"{key}_mean") for _, key in CLINICAL_METRICS]


def _argmax_per_column(rows_means: list[list[float | None]]) -> list[int]:
    """For each metric column, return the row index with the maximum value
    (None entries are skipped). Returns -1 for a column with all None."""
    n_cols = len(CLINICAL_METRICS)
    best_idx = [-1] * n_cols
    best_val = [-1.0] * n_cols
    for i, row in enumerate(rows_means):
        for j, m in enumerate(row):
            if m is not None and m > best_val[j]:
                best_val[j] = m
                best_idx[j] = i
    return best_idx


def _clinical_cells(s: dict | None, bold_mask: list[bool] | None = None) -> list[str]:
    """Build the metric cells for one row. If bold_mask is given (list of bool
    aligned with CLINICAL_METRICS), the True positions get $\\mathbf{...}$."""
    if s is None:
        return ["--"] * len(CLINICAL_METRICS)
    means = _clinical_row_means(s)
    stds  = [s.get(f"{key}_std") for _, key in CLINICAL_METRICS]
    cells = []
    for i, (m, sd) in enumerate(zip(means, stds)):
        text = fmt(m, sd)
        if bold_mask is not None and bold_mask[i] and m is not None:
            inner = text.strip("$")
            text = r"$\mathbf{" + inner + r"}$"
        cells.append(text)
    return cells


def table_different_models(fs: str = "top300", caption_fs: str = "top-300 inclusive") -> str:
    """Table A — replicates the slide's Different Models layout:
    LR + Transformer at a single feature set, with Sens/Spec/PPV/NPV +
    AUROC/F1/AUPRC (all read from the balanced-test block)."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Different Models --- Logistic Regression and Transformer on the "
                 rf"{latex_escape(caption_fs)} feature set. All metrics are computed on a "
                 r"class-balanced test subsample (paper-comparable). Values are mean $\pm$ "
                 r"standard deviation over 10-fold stratified cross-validation.}")
    lines.append(rf"\label{{tab:table1_different_models}}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * len(CLINICAL_METRICS) + r"}")
    lines.append(r"\toprule")
    head = " & ".join(label for label, _ in CLINICAL_METRICS)
    lines.append(rf"Model & {head} \\")
    lines.append(r"\midrule")
    # Collect all rows first to compute per-column argmax
    row_data = [(m, load_full(m, fs)) for m in CLINICAL_MODELS]
    rows_means = [_clinical_row_means(s) for _, s in row_data]
    best_idx = _argmax_per_column(rows_means)
    for i, (m, s) in enumerate(row_data):
        mask = [best_idx[j] == i for j in range(len(CLINICAL_METRICS))]
        cells = _clinical_cells(s, bold_mask=mask)
        lines.append(f"{latex_escape(m)} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def table_rick_scaling() -> str:
    """Table B — Different # of features on Rick's expanded list, LR + Transformer.
    Long format: one row per (model, size) with the full clinical metric set."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Different number of features — Logistic Regression and "
                 r"Transformer on Rick's expanded ICD-10 list (top-50 through top-251). "
                 r"All metrics are computed on a class-balanced test subsample "
                 r"(paper-comparable). Mean $\pm$ standard deviation over 10 folds.}")
    lines.append(r"\label{tab:table2_rick_scaling}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{ll" + "c" * len(CLINICAL_METRICS) + r"}")
    lines.append(r"\toprule")
    head = " & ".join(label for label, _ in CLINICAL_METRICS)
    lines.append(rf"Model & \# features & {head} \\")
    lines.append(r"\midrule")
    # Per-column argmax is computed WITHIN each model's group, so LR and
    # Transformer each get their own row-wise winners highlighted.
    for i, m in enumerate(CLINICAL_MODELS):
        group = [(fs, load_full(m, fs)) for fs in RICK_SIZES]
        group_means = [_clinical_row_means(s) for _, s in group]
        best_idx = _argmax_per_column(group_means)
        for j, (fs, s) in enumerate(group):
            mask = [best_idx[c] == j for c in range(len(CLINICAL_METRICS))]
            cells = _clinical_cells(s, bold_mask=mask)
            mname = latex_escape(m) if j == 0 else ""
            lines.append(f"{mname} & {RICK_SIZE_LABELS[fs]} & "
                         + " & ".join(cells) + r" \\")
        if i < len(CLINICAL_MODELS) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def table_curation_compare() -> str:
    """Bonus — three-list curation comparison: Kelly limited, Rick expanded,
    inclusive top-300; LR + Transformer; clinical metric set."""
    list_options = [
        ("top300",      "Inclusive (299)"),
        ("rick_top251", "Rick expanded (251)"),
        ("kelly",       "Kelly limited (48)"),
    ]
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Curation comparison — Logistic Regression and Transformer "
                 r"across three feature curations: the OR-ranked inclusive top-300, "
                 r"Rick's expanded list (251 codes), and Kelly's limited list (48 "
                 r"codes). All metrics are computed on a class-balanced test "
                 r"subsample. Mean $\pm$ standard deviation over 10 folds.}")
    lines.append(r"\label{tab:curation_compare}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{ll" + "c" * len(CLINICAL_METRICS) + r"}")
    lines.append(r"\toprule")
    head = " & ".join(label for label, _ in CLINICAL_METRICS)
    lines.append(rf"Model & Feature list & {head} \\")
    lines.append(r"\midrule")
    # Per-column argmax is computed WITHIN each model's group, so LR and
    # Transformer each get their own row-wise winners highlighted.
    for i, m in enumerate(CLINICAL_MODELS):
        group = [(fs, lbl, load_full(m, fs)) for fs, lbl in list_options]
        group_means = [_clinical_row_means(s) for _, _, s in group]
        best_idx = _argmax_per_column(group_means)
        for j, (fs, label, s) in enumerate(group):
            mask = [best_idx[c] == j for c in range(len(CLINICAL_METRICS))]
            cells = _clinical_cells(s, bold_mask=mask)
            mname = latex_escape(m) if j == 0 else ""
            lines.append(f"{mname} & {latex_escape(label)} & "
                         + " & ".join(cells) + r" \\")
        if i < len(CLINICAL_MODELS) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def write_paper_revision_tex():
    """Write the new paper-revision tables to a SEPARATE file, never touching tables.tex."""
    content = "\n\n".join([
        "% Table 1 — Different Models (LR + Transformer at inclusive top-300)",
        table_different_models("top300", "top-300 inclusive"),
        "",
        "% Table 2 — Different # of features on Rick's expanded list",
        table_rick_scaling(),
        "",
        "% Supplementary — Curation comparison (inclusive vs Rick vs Kelly)",
        table_curation_compare(),
    ]) + "\n"
    out = OUT / "tables_paper.tex"
    out.write_text(content)
    print(f"Saved {out}")


def main():
    sections = [
        ("% Table 2 — paper-style grouped table\n" + table_paper_style_grouped("top300")),
        ("% Table — Main results at top-300 (flat, with F1*)\n" + table_main_at_fs("top300")),
        ("% Table — AUROC across feature sets\n" + table_scaling(
            "auroc", "AUROC",
            "AUROC across feature set sizes (mean $\\pm$ std over 10 folds). "
            "Best value per column is in bold.")),
        ("% Table — F1* across feature sets\n" + table_scaling(
            "f1_best", "F1*",
            "F1 at the F1-optimal threshold (F1$^{*}$) across feature set sizes "
            "(mean $\\pm$ std over 10 folds). Best value per column is in bold.")),
        ("% Table — Threshold calibration at top-300\n" + table_calibration("top300")),
    ]
    tables_tex = "\n\n".join(sections) + "\n"
    (OUT / "tables.tex").write_text(tables_tex)
    print(f"Saved {OUT / 'tables.tex'}")

    (OUT / "figures.tex").write_text(build_figures_tex())
    print(f"Saved {OUT / 'figures.tex'}")

    # Optional standalone wrapper that compiles by itself
    standalone = r"""
\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{caption}
\begin{document}

\section*{Tables}
\input{tables.tex}

\clearpage
\section*{Figures}
\input{figures.tex}

\end{document}
""".lstrip()
    (OUT / "paper_bundle.tex").write_text(standalone)
    print(f"Saved {OUT / 'paper_bundle.tex'} (standalone wrapper)")
    print("Required LaTeX packages: booktabs, graphicx, multirow, caption")

    # Paper revision tables (LR + Transformer, clinical metric set, balanced-test
    # numbers). Written to a SEPARATE .tex so the original tables.tex is untouched.
    write_paper_revision_tex()


if __name__ == "__main__":
    main()
