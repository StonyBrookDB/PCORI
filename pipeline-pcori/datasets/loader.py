#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------- IO ----------------------
def _read_table(base: Path) -> pd.DataFrame:
    pqt = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")
    if pqt.exists():
        return pd.read_parquet(pqt)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Neither {pqt.name} nor {csv.name} exists in {base.parent}")

def load_spec(spec_path: Path) -> dict:
    return json.loads(Path(spec_path).read_text(encoding="utf-8"))

def load_middleware(dataset_dir: Path):
    enc = _read_table(dataset_dir / "encounters")
    ef  = _read_table(dataset_dir / "edges_encounter_feature")
    pe  = _read_table(dataset_dir / "edges_patient_encounter")
    features = pd.read_csv(dataset_dir / "features.csv")
    labels   = pd.read_csv(dataset_dir / "labels.csv") if (dataset_dir / "labels.csv").exists() else None
    cohort = None
    cp = dataset_dir / "cohort_run.json"
    if cp.exists():
        cohort = json.loads(cp.read_text(encoding="utf-8"))
    return enc, ef, pe, features, labels, cohort


# ---------------------- Feature/column index ----------------------
def build_feature_index(spec: dict, features: pd.DataFrame):
    """based on layout generate feature_id -> cloumn index(0..D-1),,, feature_id may not same as column indax"""
    layout = spec["layout"]
    feat_sorted = features.sort_values(["group", "feature_id"]).copy()
    start_by_group = {g: int(r[0]) for g, r in layout.items()}
    col_index = np.empty(len(feat_sorted), dtype=int)
    for g, sub in feat_sorted.groupby("group", sort=False):
        start = start_by_group[g]
        col_index[sub.index] = np.arange(start, start + len(sub))
    feat_sorted["col_index"] = col_index
    fid2col = dict(zip(
        feat_sorted["feature_id"].astype(int).tolist(),
        feat_sorted["col_index"].astype(int).tolist()
    ))
    total_dim = int(spec["total_dim"])
    return fid2col, total_dim


# ---------------------- encounter×D matrix ----------------------
def make_encounter_matrix(ef: pd.DataFrame, fid2col: dict, total_dim: int) -> pd.DataFrame:
    ef2 = ef.copy()
    # fill NaN，to 0，here
    if "value" in ef2.columns:
        ef2["value"] = pd.to_numeric(ef2["value"], errors="coerce").fillna(0.0)

    ef2["col"] = ef2["feature_id"].map(fid2col)
    ef2 = ef2.dropna(subset=["col"])
    ef2["col"] = ef2["col"].astype(int)

    agg = ef2.groupby(["encounter_id", "col"], as_index=False)["value"].sum()
    wide = agg.pivot(index="encounter_id", columns="col", values="value")
    # Ensure that columns 0..D-1 Complete; missing -> 0
    wide = wide.reindex(columns=np.arange(total_dim), fill_value=0.0)
    # if still NaN，empty 0
    wide = wide.fillna(0.0).astype(np.float32)
    return wide


# ---------------------- Construct (T,D) sequence ----------------------
def build_sequences(enc: pd.DataFrame, wide: pd.DataFrame, spec: dict):
    T = int(spec["sequence"]["seq_len"])
    D = int(spec["total_dim"])
    policy = spec["sequence"].get("padding_policy", "repeat_last")

    enc_local = enc[["encounter_id", "patient_id", "t_index"]].copy()

    # # If any encounter is missing in "wide", fill it with zero vectors
    missing_ids = set(enc_local["encounter_id"]) - set(wide.index)
    if missing_ids:
        add = pd.DataFrame(np.zeros((len(missing_ids), D), dtype=np.float32),
                           index=list(missing_ids), columns=np.arange(D))
        wide = pd.concat([wide, add], axis=0)

    enc_vec = wide.loc[enc_local["encounter_id"].astype(str)].reset_index(drop=True)
    enc_all = pd.concat([enc_local.reset_index(drop=True), enc_vec], axis=1)

    seqs, order = {}, []
    for pid, sub in enc_all.groupby("patient_id"):
        sub = sub.sort_values("t_index")
        X = sub.iloc[:, 3:].to_numpy(np.float32)  # Vector columns start from column 4
        if X.shape[0] >= T:
            X = X[:T]
        else:
            if policy == "repeat_last" and X.shape[0] > 0:
                last = X[-1:, :]
                pad = np.repeat(last, T - X.shape[0], axis=0)
            else:
                pad = np.zeros((T - X.shape[0], D), dtype=np.float32)
            X = np.vstack([X, pad])
        seqs[pid] = X
        order.append(pid)
    return seqs, order, T, D


# ---------------------- high level API ----------------------
def load_dataset(dataset_dir: str, spec_path: str):
    ddir = Path(dataset_dir)
    spec = load_spec(Path(spec_path))
    enc, ef, pe, features, labels, cohort = load_middleware(ddir)
    fid2col, D = build_feature_index(spec, features)
    wide = make_encounter_matrix(ef, fid2col, D)
    seqs, order, T, D = build_sequences(enc, wide, spec)

    # Patient-level segmentation
    if cohort and "splits" in cohort:
        train_ids = cohort["splits"]["train"]
        val_ids   = cohort["splits"]["val"]
        test_ids  = cohort["splits"]["test"]
    else:
        rng = np.random.default_rng(2024)
        pids = np.array(order)
        idx = rng.permutation(len(pids))
        n_train = int(0.8 * len(pids)); n_val = int(0.1 * len(pids))
        train_ids = pids[idx[:n_train]].tolist()
        val_ids   = pids[idx[n_train:n_train+n_val]].tolist()
        test_ids  = pids[idx[n_train+n_val:]].tolist()

    # Label
    lab_map = {}
    if labels is not None:
        lab_map = dict(zip(labels["patient_id"].astype(str), labels["label"].astype(int)))

    def make_split(ids):
        X = np.stack([seqs[pid] for pid in ids], axis=0).astype(np.float32)  # (N,T,D)
        # backup again：NaN / +Inf / -Inf into 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.array([lab_map.get(pid, 0) for pid in ids], dtype=np.float32)
        return X, y

    X_train, y_train = make_split(train_ids)
    X_val,   y_val   = make_split(val_ids)
    X_test,  y_test  = make_split(test_ids)

    return {
        "spec": spec,
        "X_train": X_train, "y_train": y_train, "ids_train": train_ids,
        "X_val":   X_val,   "y_val":   y_val,   "ids_val":   val_ids,
        "X_test":  X_test,  "y_test":  y_test,  "ids_test":  test_ids,
    }


# ---------------------- CLI ----------------------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--spec", type=str, required=True)
    args = ap.parse_args()
    out = load_dataset(args.dataset, args.spec)
    print("Shapes:",
          "train", out["X_train"].shape,
          "val", out["X_val"].shape,
          "test", out["X_test"].shape,
          "| D =", out["spec"]["total_dim"],
          "T =", out["spec"]["sequence"]["seq_len"])

if __name__ == "__main__":
    _cli()

