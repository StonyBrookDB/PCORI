#!/usr/bin/env python3
"""
================================================================================
NTK-Inspired Early Sensitivity Scoring for Diagnosis Features
================================================================================

METHOD OVERVIEW:
----------------
This script ranks diagnosis codes using a Neural Tangent Kernel (NTK)-inspired
sensitivity score computed at model initialization, WITHOUT any training.

The key insight from NTK theory is that for wide neural networks, the gradient
at initialization captures which features will be most influential during training.
We adapt this to logistic regression with recurrence-weighted features.

FEATURE CONSTRUCTION:
---------------------
For each patient p and diagnosis d:
  f_d(p) = number of DISTINCT encounters where diagnosis d appears

Recurrence weight (captures repeated diagnoses across encounters):
  w_d(p) = log(1 + f_d(p))

For each encounter e of patient p:
  x_e[d]  = 1 if diagnosis d appears in encounter e, else 0
  x'_e[d] = x_e[d] * w_d(p)  (recurrence-weighted feature)

SENSITIVITY SCORE:
------------------
At initialization with no learned weights:
  - r = positive label rate (from unique patients)
  - bias = log(r / (1 - r))  [initial bias to match class prior]
  - p = sigmoid(bias)        [predicted probability at init]

For each encounter with label y and weighted feature x'_e[d]:
  gradient_contribution = (p - y) * x'_e[d]

Per-diagnosis scores (accumulated over all encounters):
  score_abs[d] = Σ |gradient_contribution|
  score_sq[d]  = Σ (gradient_contribution)²

Diagnoses with higher scores are more sensitive at initialization and are
expected to be more influential for learning.

SCALABILITY:
------------
  - Two-pass algorithm with in-memory data structures
  - Pass 1: Compute f_d(p) for all (patient, diagnosis) pairs in a dict
  - Pass 2: Stream encounters, lookup weights, accumulate scores
  - Deduplication is chunk-local only (no global sets to avoid OOM)
  - Memory usage: O(unique patients) + O(unique (patient, diagnosis) pairs)

================================================================================
"""

import argparse
import os
import sys
import math
import logging
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np


# Global logger
logger = None


def setup_logging(output_dir: str):
    """Setup logging to both console and file."""
    global logger

    # Create logger
    logger = logging.getLogger("ntk_sensitivity")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers

    # Formatter with timestamp
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = os.path.join(output_dir, "ntk_inspired_sensitivity.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return log_file


def log(msg: str):
    """Log message to both console and file."""
    if logger is not None:
        logger.info(msg)
    else:
        # Fallback if logger not yet initialized
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def main():
    parser = argparse.ArgumentParser(
        description="NTK-inspired sensitivity scoring for diagnosis features."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input CSV (PATIENT_ID, ENCOUNTER_ID, ICD10_3Digits, LABEL)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--chunksize", type=int, default=500000,
        help="Chunk size for reading CSV (default: 500000)"
    )
    parser.add_argument(
        "--min_patients", type=int, default=50,
        help="Minimum encounters with diagnosis to include (default: 50)"
    )
    parser.add_argument(
        "--topk", type=int, default=5000,
        help="Number of top diagnoses to output (default: 5000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Create output directory first (needed for log file)
    os.makedirs(args.output, exist_ok=True)

    # Setup logging to both console and file
    log_file = setup_logging(args.output)

    log("=" * 70)
    log("NTK-INSPIRED SENSITIVITY SCORING (In-Memory Version)")
    log("=" * 70)
    log(f"Input file:    {args.input}")
    log(f"Output dir:    {args.output}")
    log(f"Log file:      {log_file}")
    log(f"Chunk size:    {args.chunksize:,}")
    log(f"Min encounters:{args.min_patients}")
    log(f"Top K:         {args.topk}")
    log(f"Seed:          {args.seed}")
    log("=" * 70)

    # =========================================================================
    # PHASE 1: Compute f_d(p) = distinct encounter counts per (patient, diagnosis)
    # =========================================================================
    log("\n[PHASE 1] Computing f_d(p) = encounter counts per (patient, diagnosis)...")

    # In-memory data structures
    # fdp[(patient_id, diagnosis)] = count of distinct encounters
    fdp = defaultdict(int)

    # Patient -> Label mapping
    patient_labels = {}

    total_rows = 0
    rows_kept = 0

    reader = pd.read_csv(
        args.input,
        dtype=str,
        chunksize=args.chunksize,
        usecols=["PATIENT_ID", "ENCOUNTER_ID", "ICD10_3Digits", "LABEL"]
    )

    for chunk_num, chunk in enumerate(reader):
        total_rows += len(chunk)

        # Drop missing
        chunk = chunk.dropna(subset=["PATIENT_ID", "ENCOUNTER_ID", "ICD10_3Digits"])

        # Deduplicate within chunk (no cross-chunk dedup to avoid OOM)
        chunk = chunk.drop_duplicates(subset=["PATIENT_ID", "ENCOUNTER_ID", "ICD10_3Digits"])
        rows_kept += len(chunk)

        # Collect patient labels (vectorized)
        label_chunk = chunk[["PATIENT_ID", "LABEL"]].drop_duplicates()
        for _, row in label_chunk.iterrows():
            pid = row["PATIENT_ID"]
            label_val = row["LABEL"]
            if pid not in patient_labels and pd.notna(label_val):
                try:
                    patient_labels[pid] = int(float(label_val))
                except (ValueError, TypeError):
                    pass

        # Increment f_d(p) for each (patient, diagnosis) pair in this chunk
        # Group by (PATIENT_ID, ICD10_3Digits) and count unique encounters
        for (pid, diag), grp in chunk.groupby(["PATIENT_ID", "ICD10_3Digits"]):
            fdp[(pid, diag)] += grp["ENCOUNTER_ID"].nunique()

        if (chunk_num + 1) % 20 == 0 or chunk_num == 0:
            log(f"  Chunk {chunk_num + 1}: {total_rows:,} rows, "
                f"{rows_kept:,} after dedup, {len(patient_labels):,} patients")

    log(f"\n  Phase 1 complete:")
    log(f"    Total rows read:       {total_rows:,}")
    log(f"    Rows after chunk dedup:{rows_kept:,}")
    log(f"    Unique (p,d) pairs:    {len(fdp):,}")
    log(f"    Unique patients:       {len(patient_labels):,}")

    # Compute positive rate from unique patients
    n_pos = sum(1 for v in patient_labels.values() if v == 1)
    n_neg = sum(1 for v in patient_labels.values() if v == 0)
    n_total_patients = n_pos + n_neg

    if n_pos == 0 or n_neg == 0:
        log("ERROR: Need patients in both label groups!")
        sys.exit(1)

    r = n_pos / n_total_patients
    bias = math.log(r / (1 - r))
    p_init = sigmoid(bias)

    log(f"    Label=1 patients:      {n_pos:,}")
    log(f"    Label=0 patients:      {n_neg:,}")
    log(f"    Positive rate r:       {r:.6f}")
    log(f"    Initial bias:          {bias:.6f}")
    log(f"    p_init = σ(bias):      {p_init:.6f}")

    # =========================================================================
    # PHASE 2: Stream encounters and compute sensitivity scores
    # =========================================================================
    log("\n[PHASE 2] Computing sensitivity scores...")

    # Accumulators per diagnosis
    score_abs = defaultdict(float)
    score_sq = defaultdict(float)
    n_encounters_with_dx = defaultdict(int)

    total_rows = 0
    encounters_processed = 0

    reader = pd.read_csv(
        args.input,
        dtype=str,
        chunksize=args.chunksize,
        usecols=["PATIENT_ID", "ENCOUNTER_ID", "ICD10_3Digits", "LABEL"]
    )

    for chunk_num, chunk in enumerate(reader):
        total_rows += len(chunk)

        # Drop missing and deduplicate within chunk (no cross-chunk dedup to avoid OOM)
        chunk = chunk.dropna(subset=["PATIENT_ID", "ENCOUNTER_ID", "ICD10_3Digits"])
        chunk = chunk.drop_duplicates(subset=["PATIENT_ID", "ENCOUNTER_ID", "ICD10_3Digits"])

        # Group by encounter to process each encounter
        encounter_groups = chunk.groupby(["PATIENT_ID", "ENCOUNTER_ID"])

        for (patient_id, encounter_id), group in encounter_groups:
            # Get label for this patient
            label_val = patient_labels.get(patient_id)
            if label_val is None:
                continue

            y = label_val
            encounters_processed += 1

            # Get unique diagnoses in this encounter
            diagnoses = group["ICD10_3Digits"].unique()

            for diag in diagnoses:
                # Lookup f_d(p) from in-memory dict
                fdp_val = fdp.get((patient_id, diag), 1)

                # Compute recurrence weight
                w_dp = math.log(1 + fdp_val)

                # x'_e[d] = 1 * w_d(p) = w_dp (since diagnosis is present)
                x_prime = w_dp

                # Gradient contribution
                grad = (p_init - y) * x_prime

                # Accumulate scores
                score_abs[diag] += abs(grad)
                score_sq[diag] += grad * grad
                n_encounters_with_dx[diag] += 1

        if (chunk_num + 1) % 20 == 0 or chunk_num == 0:
            log(f"  Chunk {chunk_num + 1}: {total_rows:,} rows, "
                f"{encounters_processed:,} encounters, "
                f"{len(score_abs):,} diagnoses")

    # Clear large data structures
    del fdp

    log(f"\n  Phase 2 complete:")
    log(f"    Total rows processed:  {total_rows:,}")
    log(f"    Encounters processed:  {encounters_processed:,}")
    log(f"    Unique diagnoses:      {len(score_abs):,}")

    # =========================================================================
    # PHASE 3: Build results and filter
    # =========================================================================
    log("\n[PHASE 3] Building results...")

    results = []
    for diag in score_abs.keys():
        n_enc = n_encounters_with_dx[diag]

        # Filter by min encounters
        if n_enc < args.min_patients:
            continue

        results.append({
            "ICD10_3Digits": diag,
            "n_encounters_with_dx": n_enc,
            "score_abs": round(score_abs[diag], 6),
            "score_sq": round(score_sq[diag], 6)
        })

    # Create DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("score_abs", ascending=False).reset_index(drop=True)

    log(f"  Diagnoses after filtering: {len(results_df):,}")

    # =========================================================================
    # PHASE 4: Write outputs
    # =========================================================================
    log("\n[PHASE 4] Writing outputs...")

    # Full results
    full_path = os.path.join(args.output, "diagnosis_ntk_sensitivity_full.csv")
    results_df.to_csv(full_path, index=False)
    log(f"  Written: {full_path} ({len(results_df):,} rows)")

    # Top K
    topk_df = results_df.head(args.topk)
    topk_path = os.path.join(args.output, "diagnosis_ntk_sensitivity_topk.csv")
    topk_df.to_csv(topk_path, index=False)
    log(f"  Written: {topk_path} ({len(topk_df):,} rows)")

    # =========================================================================
    # Print top 20 as sanity check
    # =========================================================================
    log("\n" + "=" * 70)
    log("TOP 20 DIAGNOSES BY NTK SENSITIVITY (score_abs)")
    log("=" * 70)
    top20_str = results_df.head(20).to_string(index=False)
    for line in top20_str.split('\n'):
        log(line)
    log("=" * 70)

    log("\nDONE!")


if __name__ == "__main__":
    main()


# =============================================================================
# EXAMPLE USAGE:
# =============================================================================
#
# python ntk_inspired_sensitivity.py \
#     --input /home/zihan/pcori_project/final_data/feature_selection/diagnosis.csv \
#     --output /home/zihan/pcori_project/final_data/feature_selection \
#     --chunksize 500000 \
#     --min_patients 50 \
#     --topk 5000
#
# For Slurm:
# srun python ntk_inspired_sensitivity.py \
#     --input /home/zihan/pcori_project/final_data/feature_selection/diagnosis.csv \
#     --output /home/zihan/pcori_project/final_data/feature_selection \
#     --chunksize 1000000 \
#     --min_patients 50 \
#     --topk 5000
#
