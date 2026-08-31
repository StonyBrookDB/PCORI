"""
Build patient-level ICD-10 sequence arrays for the sequence-based models
(LSTM, Bi-LSTM, Attention, Transformer).

Output (one file per feature set):
  data/sequences_top{N}.npy
    shape  = (n_patients, MAX_LEN)   dtype = int16
    row i  = tokenized sequence of patient i, chronologically ordered
    Token IDs:  0 = PAD, 1 = UNK, 2..N+1 = top-N feature codes
    Padding pads at the END of each row.
    Truncation keeps the MOST RECENT MAX_LEN tokens (older history dropped).

The patient row order matches the matrices in data/matrix_top{N}.npz so that
labels.npy applies unchanged. The order is the first-appearance order of
PATIENT_ID in diagnosis.csv, as produced by 02_build_matrix.py.

Inputs:
  data/patient_diagnosis.csv  — per-patient chronological ICD list (from
                                build_patient_table.py)
  data/diagnosis.csv          — used only to recover the matrix patient order
  data/top{N}_features.txt    — vocabulary per feature set

Usage:
  python src/04_build_sequences.py                     # build all 5 feature sets
  python src/04_build_sequences.py --features top50    # build a subset
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
MAX_LEN = 256
PAD_ID = 0
UNK_ID = 1
FEATURE_SETS = ["top50", "top100", "top150", "top200", "top300"]


def load_feature_codes(name: str) -> list[str]:
    path = DATA_DIR / f"{name}_features.txt"
    return path.read_text().strip().splitlines()


def build_patient_order() -> dict[int, int]:
    """Recover first-appearance order of patient IDs from diagnosis.csv
    to match the row ordering of matrix_top{N}.npz / labels.npy.
    """
    print("Pass 1 / 2: recovering matrix patient order from diagnosis.csv ...", flush=True)
    t0 = time.time()
    patient_idx: dict[int, int] = {}
    reader = pd.read_csv(
        DATA_DIR / "diagnosis.csv",
        usecols=["PATIENT_ID"],
        dtype={"PATIENT_ID": np.int64},
        chunksize=500_000,
    )
    for chunk_num, chunk in enumerate(reader, start=1):
        for pid in chunk["PATIENT_ID"].to_numpy():
            pid = int(pid)
            if pid not in patient_idx:
                patient_idx[pid] = len(patient_idx)
        if chunk_num % 50 == 0:
            print(f"  chunk {chunk_num:4d} | {time.time()-t0:6.0f}s | "
                  f"patients seen: {len(patient_idx):,}", flush=True)
    print(f"Matrix patient order recovered: {len(patient_idx):,} patients "
          f"({time.time()-t0:.0f}s)", flush=True)
    return patient_idx


def load_patient_sequences() -> dict[int, str]:
    """Load patient_diagnosis.csv into {patient_id: codes_str}. ~5 GB RAM."""
    print("Pass 2 / 2: loading patient_diagnosis.csv ...", flush=True)
    t0 = time.time()
    pid_to_codes: dict[int, str] = {}
    reader = pd.read_csv(
        DATA_DIR / "patient_diagnosis.csv",
        dtype={"PATIENT_ID": np.int64, "ICD10_LIST": str},
        chunksize=200_000,
    )
    for chunk_num, chunk in enumerate(reader, start=1):
        for pid, codes in zip(chunk["PATIENT_ID"].to_numpy(), chunk["ICD10_LIST"].to_numpy()):
            pid_to_codes[int(pid)] = "" if (codes is None or codes != codes) else str(codes)
        if chunk_num % 20 == 0:
            print(f"  chunk {chunk_num:4d} | {time.time()-t0:6.0f}s | "
                  f"loaded: {len(pid_to_codes):,}", flush=True)
    print(f"Patient sequences loaded: {len(pid_to_codes):,} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return pid_to_codes


def build_one_feature_set(
    feature_name: str,
    patient_idx: dict[int, int],
    pid_to_codes: dict[int, str],
):
    print(f"\nBuilding sequences for {feature_name} ...", flush=True)
    t0 = time.time()

    codes = load_feature_codes(feature_name)
    code_to_id = {c: i + 2 for i, c in enumerate(codes)}   # reserve 0=PAD, 1=UNK
    vocab_size = len(codes) + 2
    print(f"  vocab_size = {vocab_size}  (codes={len(codes)} + PAD + UNK)", flush=True)

    n_patients = len(patient_idx)
    sequences = np.zeros((n_patients, MAX_LEN), dtype=np.int16)

    n_nonempty = 0
    n_truncated = 0
    seq_lengths = []

    for pid, row in patient_idx.items():
        codes_str = pid_to_codes.get(pid, "")
        if not codes_str:
            continue
        # Map codes -> token IDs, dropping codes not in this feature set
        tokens = [code_to_id[c] for c in codes_str.split() if c in code_to_id]
        if not tokens:
            continue
        # Keep most recent MAX_LEN tokens; pad at end with PAD (0)
        if len(tokens) > MAX_LEN:
            tokens = tokens[-MAX_LEN:]
            n_truncated += 1
        sequences[row, :len(tokens)] = tokens
        n_nonempty += 1
        seq_lengths.append(len(tokens))

    out_path = DATA_DIR / f"sequences_{feature_name}.npy"
    np.save(out_path, sequences)
    print(f"  non-empty sequences: {n_nonempty:,} / {n_patients:,} "
          f"({100*n_nonempty/n_patients:.1f}%)", flush=True)
    print(f"  truncated to MAX_LEN={MAX_LEN}: {n_truncated:,}", flush=True)
    if seq_lengths:
        sl = np.array(seq_lengths)
        print(f"  seq length: mean={sl.mean():.1f}  median={np.median(sl):.0f}  "
              f"p90={np.percentile(sl, 90):.0f}  max={sl.max()}", flush=True)
    print(f"  saved {out_path}  shape={sequences.shape}  dtype={sequences.dtype}  "
          f"({time.time()-t0:.0f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", default=FEATURE_SETS,
                        help="Feature set names; each reads data/{name}_features.txt")
    args = parser.parse_args()
    for fs in args.features:
        if not (DATA_DIR / f"{fs}_features.txt").exists():
            parser.error(f"Feature list not found: data/{fs}_features.txt")

    # Heavy one-time passes shared across all feature sets
    patient_idx = build_patient_order()
    pid_to_codes = load_patient_sequences()

    for fs in args.features:
        build_one_feature_set(fs, patient_idx, pid_to_codes)


if __name__ == "__main__":
    main()
