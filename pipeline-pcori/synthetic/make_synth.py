#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def _save_parquet_or_csv(df, path_base: Path):
    """Try to save Parquet; fallback to CSV if pyarrow/fastparquet missing."""
    try:
        df.to_parquet(path_base.with_suffix(".parquet"), index=False)
    except Exception:
        df.to_csv(path_base.with_suffix(".csv"), index=False)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="./data/synth")
    p.add_argument("--n_patients", type=int, default=240)
    p.add_argument("--seq_len", type=int, default=5)
    p.add_argument("--min_gap_days", type=int, default=14)
    p.add_argument("--history_window_days", type=int, default=365)
    p.add_argument("--pos_rate", type=float, default=0.18)
    p.add_argument("--diag", type=int, default=50)
    p.add_argument("--lab", type=int, default=40)
    p.add_argument("--demo", type=int, default=3)
    p.add_argument("--clin", type=int, default=30)
    p.add_argument("--med", type=int, default=20)
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    today = pd.Timestamp.utcnow().date()

    # features
    groups = [("diag", args.diag), ("lab", args.lab), ("demo", args.demo), ("clin", args.clin), ("med", args.med)]
    rows = []; fid = 0
    for g, n in groups:
        for i in range(n):
            rows.append({"feature_id": fid, "name": f"{g}_{i:03d}", "group": g}); fid += 1
    features = pd.DataFrame(rows); features.to_csv(out_dir / "features.csv", index=False)

    # labels
    pids = [f"P{p:06d}" for p in range(args.n_patients)]
    labels = (rng.random(args.n_patients) < args.pos_rate).astype(int)
    pd.DataFrame({"patient_id": pids, "label": labels}).to_csv(out_dir / "labels.csv", index=False)

    # ranges
    offset = 0; ranges = {}
    for g, n in groups:
        ranges[g] = (offset, n); offset += n

    enc_rows, pe_rows, ef_rows = [], [], []
    for pid, y in zip(pids, labels):
        target_date = today - pd.Timedelta(days=int(rng.integers(1, 30)))
        days_back = rng.integers(args.min_gap_days, args.history_window_days + 1, size=args.seq_len); days_back.sort()
        age = int(rng.integers(18, 85)); sex = int(rng.integers(0, 2)); race = int(rng.integers(0, 5))
        for t_idx, db in enumerate(days_back):
            ts = target_date - pd.Timedelta(days=int(db))
            enc_id = f"E_{pid}_{t_idx:02d}"
            enc_rows.append({"encounter_id": enc_id, "patient_id": pid, "t_index": int(t_idx),
                             "timestamp": ts.isoformat(), "target_date": target_date.isoformat()})
            pe_rows.append({"patient_id": pid, "encounter_id": enc_id})
            k = {"diag": int(rng.integers(2, 6)), "lab": int(rng.integers(2, 6)),
                 "clin": int(rng.integers(1, 5)), "med": int(rng.integers(1, 4))}
            if y == 1:
                k["diag"] += int(rng.integers(1, 3)); k["med"] += int(rng.integers(1, 2)); k["clin"] += int(rng.integers(1, 2))
            def pick_ids(start, count, kk): 
                return rng.choice(np.arange(start, start+count), size=kk, replace=False).tolist()
            all_ids = []
            ds, dn = ranges["diag"]; all_ids += pick_ids(ds, dn, k["diag"])
            ls, ln = ranges["lab"];  all_ids += pick_ids(ls, ln, k["lab"])
            cs, cn = ranges["clin"]; all_ids += pick_ids(cs, cn, k["clin"])
            ms, mn = ranges["med"];  all_ids += pick_ids(ms, mn, k["med"])
            for fid in all_ids:
                val = float(np.round(rng.uniform(0.5, 5.0), 3))
                ef_rows.append({"encounter_id": enc_id, "feature_id": int(fid), "value": val})
            if t_idx == 0 and args.demo >= 3:
                ds, dn = ranges["demo"]
                ef_rows.append({"encounter_id": enc_id, "feature_id": int(ds+0), "value": float(age)})
                ef_rows.append({"encounter_id": enc_id, "feature_id": int(ds+1), "value": float(sex)})
                ef_rows.append({"encounter_id": enc_id, "feature_id": int(ds+2), "value": float(race)})
    encounters = pd.DataFrame(enc_rows); pe = pd.DataFrame(pe_rows).drop_duplicates(); ef = pd.DataFrame(ef_rows)

    _save_parquet_or_csv(encounters, out_dir / "encounters")
    _save_parquet_or_csv(pe, out_dir / "edges_patient_encounter")
    _save_parquet_or_csv(ef, out_dir / "edges_encounter_feature")

    # splits
    rng_local = np.random.default_rng(2024); n = len(pids); idx = rng_local.permutation(n)
    n_train = int(n * 0.8); n_val = int(n * 0.1)
    splits = {"train": [pids[i] for i in idx[:n_train].tolist()],
              "val":   [pids[i] for i in idx[n_train:n_train+n_val].tolist()],
              "test":  [pids[i] for i in idx[n_train+n_val:].tolist()]}
    cohort = {"generated_at": datetime.utcnow().isoformat(),
              "params": {"n_patients": args.n_patients, "seq_len": args.seq_len,
                         "min_gap_days": args.min_gap_days, "history_window_days": args.history_window_days},
              "splits": splits}
    with open(out_dir / "cohort_run.json", "w") as f: json.dump(cohort, f, indent=2)
    print(f"[OK] Wrote synthetic middleware to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
