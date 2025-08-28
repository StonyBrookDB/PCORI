#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
"""
tools/collect_runs.py

output format：
  <runs_root>/sum_eval-<timestamp>/

outputs（write in out_dir）：
  - summary.csv
  - summary.json
  - summary.md
  - manifest.json   # Record input parameters, write time, source runs_root, etc.

useage：
  python tools/collect_runs.py --runs_root ./runs
  # outputdir：
  python tools/collect_runs.py --runs_root ./runs --out_dir ./runs/my_eval_2025_0814

Optional parameters：
  --sort_by val_auroc|val_auprc|test_auroc|...  (default val_auroc)
  --ascending                                  (default False，descending)
  --stamp YYYYmmdd-HHMMSS                      (Default: local time; used for out_dir naming)
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd


def _read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _read_config(path: Path):
    txt = _read_text(path)
    if not txt:
        return {}
    # YAML or JSON
    try:
        import yaml
        return yaml.safe_load(txt)
    except Exception:
        try:
            return json.loads(txt)
        except Exception:
            return {}


def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _try_parse_timestamp(name: str):
    # 目录名形如：2025-08-14_06-52-42
    try:
        return datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
    except Exception:
        return None


def collect_one(run_dir: Path) -> dict:
    cfg = _read_config(run_dir / "config_resolved.yaml")
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        return None
    try:
        metrics = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception:
        return None

    row = {}
    row["run_dir"] = str(run_dir.resolve())
    ts = _try_parse_timestamp(run_dir.name)
    row["timestamp"] = ts.isoformat() if ts else run_dir.name

    # config
    row["model"] = _safe_get(cfg, "model", default=None)
    row["seed"] = _safe_get(cfg, "seed", default=None)
    row["dataset"] = _safe_get(cfg, "dataset", default=None)
    row["spec"] = _safe_get(cfg, "spec", default=None)

    # counts
    row["n_train"] = _safe_get(metrics, "n_train", default=None)
    row["n_val"]   = _safe_get(metrics, "n_val", default=None)
    row["n_test"]  = _safe_get(metrics, "n_test", default=None)

    # flatten val/test metrics
    for split in ["val", "test"]:
        md = _safe_get(metrics, split, default={}) or {}
        row[f"{split}_auroc"] = md.get("auroc", None)
        row[f"{split}_auprc"] = md.get("auprc", None)
        row[f"{split}_f1_at_0_5"] = md.get("f1_at_0_5", None)
        row[f"{split}_f1_best"] = md.get("f1_best", None)
        row[f"{split}_best_threshold"] = md.get("best_threshold", None)

    # optional: LSTM hparams
    if row["model"] == "lstm":
        hp = _safe_get(cfg, "lstm_hparams", default={}) or {}
        for k in ["epochs", "lr", "batch_size", "hidden_size", "num_layers", "dropout", "bidirectional", "device", "amp"]:
            row[f"hparam_{k}"] = hp.get(k, None)

    return row


def collect_runs(runs_root: Path) -> pd.DataFrame:
    rows = []
    if not runs_root.exists():
        return pd.DataFrame()
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        one = collect_one(d)
        if one:
            rows.append(one)
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame, out_dir: Path, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    manifest_path = out_dir / "manifest.json"

    # 1) main table
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    # 2) Markdown（前 30 行）
    cols = [
        "timestamp", "model", "n_train", "n_val", "n_test",
        "val_auroc", "val_auprc", "val_f1_best",
        "test_auroc", "test_auprc", "test_f1_best", "run_dir"
    ]
    show = [c for c in cols if c in df.columns]
    head = df[show].head(30)

    lines = [
        "| " + " | ".join(show) + " |",
        "| " + " | ".join(["---"] * len(show)) + " |"
    ]
    for _, r in head.iterrows():
        lines.append("| " + " | ".join([str(r[c]) for c in show]) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # 3) manifest (Record input parameters/time/output path)
    meta_out = dict(meta)
    meta_out.update({
        "written_at": datetime.now().isoformat(),
        "outputs": {
            "csv": str(csv_path.resolve()),
            "json": str(json_path.resolve()),
            "md": str(md_path.resolve())
        }
    })
    manifest_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    print("[OK] Wrote summary folder:", out_dir)
    print(" -", csv_path)
    print(" -", json_path)
    print(" -", md_path)
    print(" -", manifest_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="./runs")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Custom summary output directory；defult runs_root/sum_eval-<timestamp>/")
    ap.add_argument("--sort_by", type=str, default="val_auroc")
    ap.add_argument("--ascending", action="store_true", help="Descending; add option can change to Ascending")
    ap.add_argument("--stamp", type=str, default=None,
                    help="use for out_dir timestamp（YYYYmmdd-HHMMSS），default local time")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    df = collect_runs(runs_root)
    if df.empty:
        print("[WARN] No runs found at", runs_root)
        return

    # if rank
    if args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=args.ascending)

    # runs_root/sum_eval-<timestamp>/
    stamp = args.stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (runs_root / f"sum_eval-{stamp}")

    write_outputs(df, out_dir, meta={
        "runs_root": str(runs_root.resolve()),
        "sort_by": args.sort_by,
        "ascending": bool(args.ascending),
        "stamp": stamp,
        "out_dir": str(out_dir.resolve())
    })

    # short result list
    cols = [c for c in ["timestamp","model","val_auroc","val_auprc","test_auroc","test_auprc"] if c in df.columns]
    print(df[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
