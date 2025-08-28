#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
import argparse, json, hashlib
from pathlib import Path
import pandas as pd
import numpy as np

def _read_table(base: Path):
    """Read a table that may exist as .parquet or .csv (prefer Parquet)."""
    pqt = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")
    if pqt.exists():
        return pd.read_parquet(pqt)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Neither {pqt.name} nor {csv.name} exists in {base.parent}")

def _md5_of_file(p: Path, block=1<<20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

def build_spec(dataset_dir: Path, out_path: Path, seq_len:int=None, history_window_days:int=None,
               min_gap_days:int=None, padding_policy:str="repeat_last", validate:bool=True):
    # ---------- Load core tables
    enc = _read_table(dataset_dir / "encounters")
    feat = pd.read_csv(dataset_dir / "features.csv")
    pe = _read_table(dataset_dir / "edges_patient_encounter")
    ef = _read_table(dataset_dir / "edges_encounter_feature")

    # ---------- Basic integrity checks
    required_enc_cols = {"patient_id","encounter_id","t_index"}
    missing = required_enc_cols - set(enc.columns)
    if missing:
        raise ValueError(f"encounters is missing columns: {missing}")
    if "feature_id" not in ef.columns or "encounter_id" not in ef.columns:
        raise ValueError("edges_encounter_feature must contain encounter_id, feature_id")
    if "group" not in feat.columns or "feature_id" not in feat.columns:
        raise ValueError("features.csv must contain feature_id, group")
    # unique feature ids
    if feat["feature_id"].duplicated().any():
        dups = feat[feat["feature_id"].duplicated()]["feature_id"].unique().tolist()
        raise ValueError(f"features.csv has duplicated feature_id(s): {dups[:5]}...")

    # t_index range by patient
    seq_stat = enc.groupby("patient_id")["t_index"].agg(["min","max","count"]).reset_index()
    min_t = int(seq_stat["min"].min())
    max_t = int(seq_stat["max"].max())
    per_patient_counts = seq_stat["count"].unique().tolist()

    # infer seq_len if not provided
    inferred_seq_len = int(seq_stat["count"].mode().iloc[0]) if not per_patient_counts else int(max(per_patient_counts))
    if seq_len is None: seq_len = inferred_seq_len

    # ---------- Derive group counts & layout
    feat = feat.sort_values(["group","feature_id"]).reset_index(drop=True)
    groups = []
    cursor = 0
    for g, sub in feat.groupby("group", sort=False):
        count = int(len(sub))
        groups.append({
            "name": g,
            "count": count,
            "start": cursor,
            "end": cursor + count,
            "dtype": "float32"  # keep simple; loaders can refine per-group later
        })
        cursor += count
    total_dim = int(cursor)

    layout = {row["name"]: [int(row["start"]), int(row["end"])] for row in groups}
    groups_dict = {row["name"]: {"count": int(row["end"]-row["start"]), "dtype": row["dtype"]} for row in groups}

    # ---------- Graph schema (fixed for this project)
    node_types = ["patient","encounter","feature"]
    edge_types = [["patient","has","encounter"], ["encounter","has","feature"]]
    edge_attr_semantics = {"encounter-has-feature":"value"}  # numeric value semantics

    # ---------- Code maps (for dx/med shortcuts if present)
    counts_by_group = feat.groupby("group").size().to_dict()
    dx_size  = int(counts_by_group.get("diag", 0))
    med_size = int(counts_by_group.get("med", 0))
    med_offset = dx_size  # eliminate hard-coded +391; use dynamic offset

    # ---------- Optional stats for continuous groups (simple, loader can use later)
    # We do a light inference: if a group's values look mostly integer 0/1 in EF, mark "binary"; else "numeric".
    # (No heavy scan: sample 100k rows for speed).
    sample = ef[["feature_id","value"]].sample(n=min(100000, len(ef)), random_state=17) if len(ef) > 0 else ef
    fid2group = feat.set_index("feature_id")["group"].to_dict()
    group_value_kinds = {g:"numeric" for g in counts_by_group.keys()}
    if len(sample) > 0:
        # group small heuristic
        samp = sample.copy()
        samp["group"] = samp["feature_id"].map(fid2group)
        by_g = samp.groupby("group")["value"]
        for g, s in by_g:
            if len(s) == 0: 
                continue
            frac_int01 = float(((s - s.astype(int)).abs() < 1e-6).mean() * (s.between(0,1).mean()))
            group_value_kinds[g] = "binary" if frac_int01 > 0.9 else "numeric"

    # ---------- Integrity summary (counts & simple hashes of input files)
    # file hashes (if csv exists)
    files = {}
    for base in ["encounters","edges_patient_encounter","edges_encounter_feature","features","labels","cohort_run"]:
        pqt = dataset_dir / f"{base}.parquet"
        csv = dataset_dir / f"{base}.csv"
        if pqt.exists():
            files[f"{base}.parquet"] = {"path": str(pqt), "md5": _md5_of_file(pqt)}
        if csv.exists():
            files[f"{base}.csv"] = {"path": str(csv), "md5": _md5_of_file(csv)}

    integrity = {
        "num_patients": int(enc["patient_id"].nunique()),
        "num_encounters": int(len(enc)),
        "num_features": int(len(feat)),
        "num_edges_patient_encounter": int(len(pe)),
        "num_edges_encounter_feature": int(len(ef)),
        "per_patient_seq_count_unique": per_patient_counts,
        "t_index_min": min_t,
        "t_index_max": max_t,
        "files": files
    }

    # ---------- Final spec
    spec = {
        "version": "1.0.0",
        "dataset_root": str(dataset_dir),
        "files_present": list(files.keys()),
        "sequence": {
            "seq_len": int(seq_len),
            "history_window_days": int(history_window_days) if history_window_days is not None else None,
            "min_gap_days": int(min_gap_days) if min_gap_days is not None else None,
            "padding_policy": padding_policy
        },
        "groups": groups_dict,
        "layout": layout,
        "total_dim": total_dim,
        "graph": {
            "node_types": node_types,
            "edge_types": edge_types,
            "edge_attr_semantics": edge_attr_semantics
        },
        "code_maps": {
            "dx_size": dx_size,
            "med_size": med_size,
            "med_offset": med_offset
        },
        "value_kind": group_value_kinds,
        "integrity_checks": integrity
    }

    # ---------- Optional validation
    if validate:
        # layout covers [0,total_dim)
        covered = []
        for name, (s,e) in layout.items():
            assert 0 <= s < e <= total_dim, f"bad layout range for {name}: {(s,e)}"
            covered.extend(list(range(s,e)))
        assert len(covered) == total_dim, "layout not fully covering total_dim (overlap or gaps)"
        # patient-level t_index must be contiguous or at least not exceed seq_len-1
        if max_t+1 != seq_len:
            # Allow minor differences but warn via print (no raise to keep flexible)
            print(f"[WARN] inferred max t_index ({max_t}) != seq_len-1 ({seq_len-1})")
        # FK checks
        enc_ids = set(enc["encounter_id"].astype(str).tolist())
        feat_ids = set(feat["feature_id"].astype(int).tolist())
        if not set(ef["encounter_id"].astype(str)).issubset(enc_ids):
            raise ValueError("edges_encounter_feature contains encounter_id not in encounters")
        if not set(ef["feature_id"].astype(int)).issubset(feat_ids):
            raise ValueError("edges_encounter_feature contains feature_id not in features")

    # ---------- Write outputs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    # also write a tiny integrity json for quick checks
    integ_path = out_path.parent / "spec_integrity.json"
    with open(integ_path, "w", encoding="utf-8") as f:
        json.dump(integrity, f, indent=2)

    print(f"[OK] FeatureSpec written to {out_path}")
    groups_str = ", ".join([f"{k}:{v['count']}" for k,v in groups_dict.items()])
    print(f"     total_dim={total_dim}, groups={{{groups_str}}}")
    return spec

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Folder containing encounters/features/edges files")
    ap.add_argument("--out", type=str, required=True, help="Output path for FeatureSpec.json")
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--history_window_days", type=int, default=None)
    ap.add_argument("--min_gap_days", type=int, default=None)
    ap.add_argument("--padding_policy", type=str, default="repeat_last")
    ap.add_argument("--no_validate", action="store_true")
    args = ap.parse_args()

    build_spec(Path(args.dataset), Path(args.out),
               seq_len=args.seq_len, history_window_days=args.history_window_days,
               min_gap_days=args.min_gap_days, padding_policy=args.padding_policy,
               validate=(not args.no_validate))

if __name__ == "__main__":
    main()
