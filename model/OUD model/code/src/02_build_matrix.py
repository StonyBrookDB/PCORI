"""
Build a patient-level sparse binary feature matrix from diagnosis.csv.

For each patient:
  - Feature vector: binary presence of each selected ICD-10 code across all encounters
  - Label: 1 if the patient has any row with LABEL=1 (i.e., any OUD encounter)

Output: data/matrix_top{N}.npz  and  data/labels_top{N}.npy

Usage:
  python src/02_build_matrix.py --features top100
  python src/02_build_matrix.py --features top200
  python src/02_build_matrix.py --features top300
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNK_SIZE = 500_000


def load_feature_list(name: str) -> list[str]:
    path = DATA_DIR / f"{name}_features.txt"
    return path.read_text().strip().splitlines()


def build_matrix(feature_name: str):
    features = load_feature_list(feature_name)
    feat_idx = {code: i for i, code in enumerate(features)}
    feat_set = set(features)
    n_features = len(features)

    print(f"Building matrix for {feature_name}: {n_features} features")

    # patient_id -> index mapping (built incrementally)
    patient_idx: dict[int, int] = {}

    # Accumulate COO data for sparse matrix
    rows, cols = [], []
    # Track which patients are OUD-positive
    oud_patients: set[int] = set()

    diag_path = DATA_DIR / "diagnosis.csv"
    chunk_num = 0
    t0 = time.time()

    reader = pd.read_csv(
        diag_path,
        dtype={"PATIENT_ID": np.int64, "ENCOUNTER_ID": np.int64,
               "ICD10_3Digits": str, "LABEL": np.int8},
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        chunk_num += 1
        if chunk_num % 50 == 0:
            elapsed = time.time() - t0
            n_pts = len(patient_idx)
            print(f"  Chunk {chunk_num:4d} | {elapsed:6.0f}s | patients seen: {n_pts:,}")

        # Collect OUD labels for ALL rows (regardless of feature set)
        oud_mask = chunk["LABEL"] == 1
        for pid in chunk.loc[oud_mask, "PATIENT_ID"].unique():
            oud_patients.add(int(pid))

        # Register all patients (even those without selected features)
        for pid in chunk["PATIENT_ID"].unique():
            pid = int(pid)
            if pid not in patient_idx:
                patient_idx[pid] = len(patient_idx)

        # Filter to selected features only
        sub = chunk[chunk["ICD10_3Digits"].isin(feat_set)][["PATIENT_ID", "ICD10_3Digits"]].drop_duplicates()
        if sub.empty:
            continue

        sub_rows = sub["PATIENT_ID"].map(patient_idx).to_numpy()
        sub_cols = sub["ICD10_3Digits"].map(feat_idx).to_numpy()
        rows.append(sub_rows)
        cols.append(sub_cols)

    n_patients = len(patient_idx)
    print(f"\nTotal patients: {n_patients:,}")
    print(f"Total OUD patients: {len(oud_patients):,}  ({100*len(oud_patients)/n_patients:.2f}%)")

    # Build sparse matrix
    all_rows = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
    all_cols = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
    data_vals = np.ones(len(all_rows), dtype=np.int8)
    X = csr_matrix((data_vals, (all_rows, all_cols)), shape=(n_patients, n_features), dtype=np.int8)
    # Binarize (a patient may appear multiple times with same code across encounters)
    X.data[:] = 1

    # Build label array in patient_idx order
    pid_order = sorted(patient_idx, key=lambda p: patient_idx[p])
    y = np.array([1 if pid in oud_patients else 0 for pid in pid_order], dtype=np.int8)

    # Save matrix
    out_matrix = DATA_DIR / f"matrix_{feature_name}.npz"
    save_npz(out_matrix, X)
    print(f"Saved matrix: {out_matrix}  shape={X.shape}")

    # Labels are feature-independent — save once as labels.npy, skip if already exists
    out_labels = DATA_DIR / "labels.npy"
    if not out_labels.exists():
        np.save(out_labels, y)
        print(f"Saved labels: {out_labels}  positive_rate={y.mean():.4f}")
    else:
        print(f"Labels already exist at {out_labels}, skipping.")
    print(f"Total time: {time.time()-t0:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="top100",
                        help="Feature set name; reads data/{name}_features.txt")
    args = parser.parse_args()
    feat_path = DATA_DIR / f"{args.features}_features.txt"
    if not feat_path.exists():
        parser.error(f"Feature list not found: {feat_path}. "
                     f"Run src/01_extract_features.py first.")
    build_matrix(args.features)


if __name__ == "__main__":
    main()
