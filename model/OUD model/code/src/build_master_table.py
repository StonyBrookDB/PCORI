"""
Build ONE master table of results: all models × all feature sets.

Default = KEY metrics only (the ones that matter for the paper):
    AUROC      — primary, threshold-independent ranking metric
    F1*        — F1 at the F1-optimal threshold (the "honest best F1")
    Calib-F1   — F1 at the prevalence-matched operating point (clinical)

Use --full to also include Precision / Recall / F1@0.5 / thr* / calib P,R,thr.
Anything not yet finished shows as '-'.

Sources per experiment:
  experiments/{slug}_{fs}/summary.json              (main: thr=0.5 + F1*)
  experiments/{slug}_{fs}/summary_calibration.json  (prevalence_matched)

Outputs:
  results/master_table_key.csv   (default)
  results/master_table.csv       (with --full)

Usage:
  python src/build_master_table.py            # key metrics
  python src/build_master_table.py --full     # everything
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model_registry import MODEL_REGISTRY

ROOT = Path(__file__).parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "results" / "master_table.csv"
OUT.parent.mkdir(exist_ok=True)

FEATURE_SETS = ["top50", "top100", "top150", "top200", "top300"]
DASH = "-"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def fmt(mean, std=None):
    if mean is None:
        return DASH
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"


def load(model_name, fs):
    """Return (main_summary | None, calib_summary | None)."""
    d = EXP / f"{slug(model_name)}_{fs}"
    main = d / "summary.json"
    calib = d / "summary_calibration.json"
    main_s = json.loads(main.read_text()) if main.exists() else None
    calib_s = json.loads(calib.read_text()) if calib.exists() else None
    return main_s, calib_s


def collect_rows():
    rows = []
    for entry in MODEL_REGISTRY:
        model_name = entry[0]
        for fs in FEATURE_SETS:
            main_s, calib_s = load(model_name, fs)

            def mget(k):
                return main_s.get(k) if main_s else None

            pm = (calib_s["strategies"]["prevalence_matched"]
                  if calib_s and "strategies" in calib_s
                  and "prevalence_matched" in calib_s["strategies"] else None)

            def pmget(k):
                return pm.get(k) if pm else None

            rows.append({
                "Model": model_name,
                "Features": fs,
                "Precision": (mget("precision_mean"), mget("precision_std")),
                "Recall": (mget("recall_mean"), mget("recall_std")),
                "F1@0.5": (mget("f1_mean"), mget("f1_std")),
                "AUROC": (mget("auroc_mean"), mget("auroc_std")),
                "F1*": (mget("f1_best_mean"), mget("f1_best_std")),
                "thr*": (mget("best_threshold_mean"), None),
                "calP": (pmget("precision_mean"), pmget("precision_std")),
                "calR": (pmget("recall_mean"), pmget("recall_std")),
                "Calib-F1": (pmget("f1_mean"), pmget("f1_std")),
                "calThr": (pmget("threshold_mean"), None),
                "status": ("main" if main_s else "") + ("+calib" if calib_s else ""),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Include all metrics (default: key metrics only)")
    args = parser.parse_args()

    rows = collect_rows()

    if args.full:
        metric_cols = ["Precision", "Recall", "F1@0.5", "AUROC",
                       "F1*", "thr*", "calP", "calR", "Calib-F1", "calThr"]
        title = "MASTER RESULTS TABLE — FULL"
        out_path = ROOT / "results" / "master_table.csv"
    else:
        metric_cols = ["AUROC", "F1*", "Calib-F1"]
        title = "MASTER RESULTS TABLE — KEY METRICS"
        out_path = ROOT / "results" / "master_table_key.csv"

    metric_w = 16
    header = f"{'Model':<20}{'Feat':<8}" + "".join(f"{c:>{metric_w}}" for c in metric_cols)
    print("=" * len(header))
    print(f"  {title}   (— = not finished yet)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    last_model = None
    for r in rows:
        if last_model is not None and r["Model"] != last_model:
            print("-" * len(header))
        last_model = r["Model"]
        line = f"{r['Model']:<20}{r['Features']:<8}"
        for c in metric_cols:
            mean, std = r[c]
            line += f"{fmt(mean, std):>{metric_w}}"
        print(line)
    print("=" * len(header))

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["Model", "Features"]
        for c in metric_cols:
            head += [c] if c in ("thr*", "calThr") else [f"{c}_mean", f"{c}_std"]
        head += ["status"]
        w.writerow(head)
        for r in rows:
            def cell(v):
                return DASH if v is None else f"{v:.4f}"
            line = [r["Model"], r["Features"]]
            for c in metric_cols:
                mean, std = r[c]
                if c in ("thr*", "calThr"):
                    line.append(cell(mean))
                else:
                    line += [cell(mean), cell(std)]
            line.append(r["status"] or DASH)
            w.writerow(line)
    print(f"\nSaved to {out_path}")

    done = sum(1 for r in rows if "main" in r["status"])
    calib = sum(1 for r in rows if "calib" in r["status"])
    print(f"Coverage: main {done}/{len(rows)}   calibration {calib}/{len(rows)}")


if __name__ == "__main__":
    main()
