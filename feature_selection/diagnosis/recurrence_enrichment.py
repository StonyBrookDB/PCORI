#!/usr/bin/env python3
"""
================================================================================
Recurrence-Aware, Label-Aware Feature Selection for Diagnosis Codes
================================================================================

METHOD OVERVIEW:
----------------
This script identifies diagnosis codes (ICD10 3-digit) whose recurrence patterns
differ most between positive (label=1) and negative (label=0) patient groups.

Key Insight:
  Patients with certain conditions may have specific diagnoses appearing across
  MULTIPLE encounters, not just once. This recurrence signal can be more
  informative than simple presence/absence.

Definitions:
  For each patient p and diagnosis d:
    f_d(p) = number of DISTINCT encounters in which diagnosis d appears

  For each diagnosis d, we compute:
    - n_patients_with_dx: count of unique patients with f_d(p) >= 1
    - sum_f_pos: sum of f_d(p) over all label=1 patients
    - sum_f_neg: sum of f_d(p) over all label=0 patients
    - mean_f_pos: sum_f_pos / N_pos_patients (total positive patients)
    - mean_f_neg: sum_f_neg / N_neg_patients (total negative patients)
    - score_diff: mean_f_pos - mean_f_neg
    - score_ratio: mean_f_pos / (mean_f_neg + 1e-9)

  Diagnoses are ranked by score_diff (descending) to find codes whose
  recurrence is most elevated in positive patients.

SCALABILITY:
------------
  - Uses chunked CSV reading (default 500K rows)
  - Hash-partitions data by diagnosis code to disk
  - Reduces each partition independently to minimize memory usage
  - Suitable for datasets with hundreds of millions of rows

OUTPUTS:
--------
  1. diagnosis_recurrence_enrichment_full.csv  - All diagnoses meeting min_patients
  2. diagnosis_recurrence_enrichment_topk.csv  - Top K by score_diff

================================================================================
"""

import argparse
import os
import sys
import hashlib
import shutil
import tempfile
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def hash_partition(diagnosis: str, n_partitions: int) -> int:
    """Deterministic hash partition for a diagnosis code."""
    return int(hashlib.md5(str(diagnosis).encode()).hexdigest(), 16) % n_partitions


def main():
    parser = argparse.ArgumentParser(
        description="Recurrence-aware feature selection for diagnosis codes."
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
        help="Minimum patients with diagnosis to include (default: 50)"
    )
    parser.add_argument(
        "--topk", type=int, default=5000,
        help="Number of top diagnoses to output (default: 5000)"
    )
    parser.add_argument(
        "--n_partitions", type=int, default=100,
        help="Number of hash partitions for disk-based processing (default: 100)"
    )

    args = parser.parse_args()

    log("=" * 70)
    log("RECURRENCE ENRICHMENT FEATURE SELECTION")
    log("=" * 70)
    log(f"Input file:    {args.input}")
    log(f"Output dir:    {args.output}")
    log(f"Chunk size:    {args.chunksize:,}")
    log(f"Min patients:  {args.min_patients}")
    log(f"Top K:         {args.topk}")
    log(f"Partitions:    {args.n_partitions}")
    log("=" * 70)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create temp directory for partitions
    temp_dir = tempfile.mkdtemp(prefix="recurrence_enrichment_")
    log(f"Temp directory: {temp_dir}")

    try:
        # =====================================================================
        # PHASE 1: Partition data to disk and collect patient labels
        # =====================================================================
        log("\n[PHASE 1] Partitioning data to disk...")

        # Patient -> Label mapping (patient-level, fits in memory)
        patient_labels = {}

        # Open partition files for writing
        partition_files = {}
        for i in range(args.n_partitions):
            path = os.path.join(temp_dir, f"partition_{i:04d}.csv")
            partition_files[i] = open(path, 'w')
            partition_files[i].write("PATIENT_ID,ENCOUNTER_ID,ICD10_3Digits\n")

        total_rows = 0
        rows_kept = 0

        reader = pd.read_csv(
            args.input,
            dtype=str,
            chunksize=args.chunksize,
            usecols=["PATIENT_ID", "ENCOUNTER_ID", "ICD10_3Digits", "LABEL"]
        )

        for chunk_num, chunk in enumerate(reader):
            chunk_start_rows = len(chunk)
            total_rows += chunk_start_rows

            # Drop rows with missing required fields
            chunk = chunk.dropna(subset=["PATIENT_ID", "ENCOUNTER_ID", "ICD10_3Digits"])

            # Deduplicate within chunk (same patient, encounter, diagnosis = 1 occurrence)
            chunk = chunk.drop_duplicates(subset=["PATIENT_ID", "ENCOUNTER_ID", "ICD10_3Digits"])

            rows_kept += len(chunk)

            # Collect patient labels
            for _, row in chunk[["PATIENT_ID", "LABEL"]].drop_duplicates().iterrows():
                pid = row["PATIENT_ID"]
                label = row["LABEL"]
                if pid not in patient_labels and pd.notna(label):
                    try:
                        patient_labels[pid] = int(float(label))
                    except (ValueError, TypeError):
                        pass

            # Write to partitions
            for _, row in chunk.iterrows():
                diag = row["ICD10_3Digits"]
                part_id = hash_partition(diag, args.n_partitions)
                partition_files[part_id].write(
                    f"{row['PATIENT_ID']},{row['ENCOUNTER_ID']},{diag}\n"
                )

            if (chunk_num + 1) % 20 == 0 or chunk_num == 0:
                log(f"  Chunk {chunk_num + 1}: {total_rows:,} rows read, "
                    f"{rows_kept:,} kept, {len(patient_labels):,} patients")

        # Close partition files
        for f in partition_files.values():
            f.close()

        log(f"\n  Phase 1 complete:")
        log(f"    Total rows read:    {total_rows:,}")
        log(f"    Rows after dedup:   {rows_kept:,}")
        log(f"    Unique patients:    {len(patient_labels):,}")

        # Count patients per label
        n_pos = sum(1 for v in patient_labels.values() if v == 1)
        n_neg = sum(1 for v in patient_labels.values() if v == 0)
        log(f"    Label=1 patients:   {n_pos:,}")
        log(f"    Label=0 patients:   {n_neg:,}")

        if n_pos == 0 or n_neg == 0:
            log("ERROR: Need patients in both label groups!")
            sys.exit(1)

        # =====================================================================
        # PHASE 2: Reduce partitions and compute statistics
        # =====================================================================
        log("\n[PHASE 2] Reducing partitions and computing statistics...")

        # diagnosis -> {n_patients, sum_f_pos, sum_f_neg}
        diagnosis_stats = defaultdict(lambda: {
            "n_patients": 0,
            "sum_f_pos": 0,
            "sum_f_neg": 0
        })

        for part_id in range(args.n_partitions):
            part_path = os.path.join(temp_dir, f"partition_{part_id:04d}.csv")

            # Skip empty partitions
            if os.path.getsize(part_path) <= 50:  # Just header
                continue

            # Read partition
            part_df = pd.read_csv(part_path, dtype=str)

            if len(part_df) == 0:
                continue

            # Deduplicate again (across chunks, same patient-encounter-diagnosis)
            part_df = part_df.drop_duplicates(
                subset=["PATIENT_ID", "ENCOUNTER_ID", "ICD10_3Digits"]
            )

            # Group by (PATIENT_ID, ICD10_3Digits) and count distinct encounters
            # This gives us f_d(p) for each (patient, diagnosis) pair
            patient_diag_counts = part_df.groupby(
                ["PATIENT_ID", "ICD10_3Digits"]
            )["ENCOUNTER_ID"].nunique().reset_index()
            patient_diag_counts.columns = ["PATIENT_ID", "ICD10_3Digits", "f_dp"]

            # Now aggregate to diagnosis level
            for diag, group in patient_diag_counts.groupby("ICD10_3Digits"):
                n_patients_dx = 0
                sum_f_pos = 0
                sum_f_neg = 0

                for _, row in group.iterrows():
                    pid = row["PATIENT_ID"]
                    f_dp = row["f_dp"]

                    if pid in patient_labels:
                        label = patient_labels[pid]
                        n_patients_dx += 1
                        if label == 1:
                            sum_f_pos += f_dp
                        else:
                            sum_f_neg += f_dp

                diagnosis_stats[diag]["n_patients"] += n_patients_dx
                diagnosis_stats[diag]["sum_f_pos"] += sum_f_pos
                diagnosis_stats[diag]["sum_f_neg"] += sum_f_neg

            if (part_id + 1) % 20 == 0 or part_id == 0:
                log(f"  Processed partition {part_id + 1}/{args.n_partitions}, "
                    f"{len(diagnosis_stats):,} diagnoses so far")

        log(f"\n  Phase 2 complete: {len(diagnosis_stats):,} unique diagnoses")

        # =====================================================================
        # PHASE 3: Compute final metrics and filter
        # =====================================================================
        log("\n[PHASE 3] Computing final metrics...")

        results = []
        for diag, stats in diagnosis_stats.items():
            n_patients = stats["n_patients"]
            sum_f_pos = stats["sum_f_pos"]
            sum_f_neg = stats["sum_f_neg"]

            # Skip diagnoses with too few patients
            if n_patients < args.min_patients:
                continue

            # Compute means (normalize by total patients in each group)
            mean_f_pos = sum_f_pos / n_pos
            mean_f_neg = sum_f_neg / n_neg

            # Compute scores
            score_diff = mean_f_pos - mean_f_neg
            score_ratio = mean_f_pos / (mean_f_neg + 1e-9)

            results.append({
                "ICD10_3Digits": diag,
                "n_patients_with_dx": n_patients,
                "sum_f_pos": sum_f_pos,
                "sum_f_neg": sum_f_neg,
                "mean_f_pos": round(mean_f_pos, 6),
                "mean_f_neg": round(mean_f_neg, 6),
                "score_diff": round(score_diff, 6),
                "score_ratio": round(score_ratio, 6)
            })

        # Create DataFrame and sort by score_diff descending
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("score_diff", ascending=False).reset_index(drop=True)

        log(f"  Diagnoses after min_patients filter: {len(results_df):,}")

        # =====================================================================
        # PHASE 4: Write outputs
        # =====================================================================
        log("\n[PHASE 4] Writing output files...")

        # Full results
        full_path = os.path.join(args.output, "diagnosis_recurrence_enrichment_full.csv")
        results_df.to_csv(full_path, index=False)
        log(f"  Written: {full_path} ({len(results_df):,} rows)")

        # Top K results
        topk_df = results_df.head(args.topk)
        topk_path = os.path.join(args.output, "diagnosis_recurrence_enrichment_topk.csv")
        topk_df.to_csv(topk_path, index=False)
        log(f"  Written: {topk_path} ({len(topk_df):,} rows)")

        # =====================================================================
        # Print top 20 as sanity check
        # =====================================================================
        log("\n" + "=" * 70)
        log("TOP 20 DIAGNOSES BY RECURRENCE ENRICHMENT (score_diff)")
        log("=" * 70)
        print(results_df.head(20).to_string(index=False))
        log("=" * 70)

        log("\nDONE!")

    finally:
        # Cleanup temp directory
        log(f"\nCleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()


# =============================================================================
# EXAMPLE USAGE:
# =============================================================================
#
# python recurrence_enrichment.py \
#     --input /home/zihan/pcori_project/final_data/feature_selection/diagnosis.csv \
#     --output /home/zihan/pcori_project/final_data/feature_selection \
#     --chunksize 500000 \
#     --min_patients 50 \
#     --topk 5000
#
# For Slurm:
# srun python recurrence_enrichment.py \
#     --input /home/zihan/pcori_project/final_data/feature_selection/diagnosis.csv \
#     --output /home/zihan/pcori_project/final_data/feature_selection \
#     --chunksize 1000000 \
#     --min_patients 50 \
#     --topk 5000
#
