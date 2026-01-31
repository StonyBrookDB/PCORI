#!/usr/bin/env python3
"""
NTK-style / early-sensitivity feature selection for laboratory tests.

Computes patient-level abnormality features and ranks labs by their
gradient-based sensitivity score at initialization (w=0).
"""

import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

INPUT_LAB_FILE = "/home/zihan/pcori_project/final_data/processed_data/encounter5/lab_test_with_patient_label.csv"
INPUT_LAB_DICT = "/home/zihan/pcori_project/final_data/HF_D_LAB_PROCEDURE.csv"
OUTPUT_DIR = "/home/zihan/pcori_project/final_data/feature_selection/labs/"

CHUNK_SIZE = 500_000

# Indicator values considered abnormal (fallback rule)
ABNORMAL_INDICATORS = {"Low", "High", "Abnormal", "Critical", "Outside limits"}


def setup_logging():
    """Configure logging to file and console."""
    log_file = os.path.join(OUTPUT_DIR, "lab_ntk_feature_selection.log")

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def determine_abnormality_vectorized(chunk):
    """
    Vectorized abnormality detection for an entire chunk.

    Rules (strict):
    1) Primary: numeric-based if all three values present
       Abnormal if NUMERIC_RESULT < NORMAL_RANGE_LOW or > NORMAL_RANGE_HIGH
    2) Fallback: indicator-based if numeric not available
       Abnormal if RESULT_INDICATOR_DESC in ABNORMAL_INDICATORS

    Returns a boolean Series.
    """
    numeric_result = chunk['NUMERIC_RESULT']
    low = chunk['NORMAL_RANGE_LOW']
    high = chunk['NORMAL_RANGE_HIGH']
    indicator = chunk['RESULT_INDICATOR_DESC']

    # Check where all three numeric values are present
    has_numeric = numeric_result.notna() & low.notna() & high.notna()

    # Numeric-based abnormality (only where has_numeric is True)
    numeric_abnormal = (numeric_result < low) | (numeric_result > high)

    # Indicator-based abnormality (fallback)
    indicator_abnormal = indicator.astype(str).str.strip().isin(ABNORMAL_INDICATORS)

    # Apply rules: use numeric where available, else use indicator
    is_abnormal = np.where(has_numeric, numeric_abnormal, indicator_abnormal)

    return pd.Series(is_abnormal, index=chunk.index, dtype=bool)


def count_file_lines(filepath):
    """Count lines in file efficiently for progress bar."""
    with open(filepath, 'rb') as f:
        return sum(1 for _ in f) - 1  # Subtract header


def process_lab_data_in_chunks(logger):
    """
    Process lab data in chunks to build patient-level abnormal flags.
    Returns a DataFrame with (PATIENT_ID, DETAIL_LAB_PROCEDURE_ID, abnormal_flag).
    """
    logger.info(f"Reading lab data in chunks from: {INPUT_LAB_FILE}")

    # Count total rows for progress bar
    logger.info("Counting total rows for progress bar...")
    total_lines = count_file_lines(INPUT_LAB_FILE)
    total_chunks = (total_lines + CHUNK_SIZE - 1) // CHUNK_SIZE
    logger.info(f"Total rows: {total_lines:,}, expected chunks: {total_chunks}")

    # Dictionary to track abnormal flags: (patient_id, lab_id) -> True/False
    # We only need to track if ANY abnormal exists, so we use a set for abnormals
    abnormal_pairs = set()  # (PATIENT_ID, DETAIL_LAB_PROCEDURE_ID) pairs with abnormal
    all_pairs = set()  # All (PATIENT_ID, DETAIL_LAB_PROCEDURE_ID) pairs seen
    patient_labels = {}  # PATIENT_ID -> LABEL

    total_rows = 0

    # Required columns
    usecols = ['PATIENT_ID', 'DETAIL_LAB_PROCEDURE_ID', 'NUMERIC_RESULT',
               'NORMAL_RANGE_LOW', 'NORMAL_RANGE_HIGH', 'RESULT_INDICATOR_DESC', 'LABEL']

    chunk_iter = pd.read_csv(INPUT_LAB_FILE, chunksize=CHUNK_SIZE, usecols=usecols,
                             low_memory=False)

    for chunk in tqdm(chunk_iter, total=total_chunks, desc="Processing chunks"):
        total_rows += len(chunk)

        # Convert numeric columns with robust parsing
        chunk['NUMERIC_RESULT'] = pd.to_numeric(chunk['NUMERIC_RESULT'], errors='coerce')
        chunk['NORMAL_RANGE_LOW'] = pd.to_numeric(chunk['NORMAL_RANGE_LOW'], errors='coerce')
        chunk['NORMAL_RANGE_HIGH'] = pd.to_numeric(chunk['NORMAL_RANGE_HIGH'], errors='coerce')

        # Determine abnormality using vectorized operations
        chunk['is_abnormal'] = determine_abnormality_vectorized(chunk)

        # Update patient labels (vectorized)
        label_df = chunk[['PATIENT_ID', 'LABEL']].drop_duplicates()
        label_df = label_df[label_df['LABEL'].notna()]
        for pid, label in zip(label_df['PATIENT_ID'], label_df['LABEL']):
            patient_labels[pid] = int(label)

        # Update pair tracking (vectorized using tuples from arrays)
        pids = chunk['PATIENT_ID'].values
        lab_ids = chunk['DETAIL_LAB_PROCEDURE_ID'].values
        is_abnormal = chunk['is_abnormal'].values

        # Add all pairs
        chunk_pairs = set(zip(pids, lab_ids))
        all_pairs.update(chunk_pairs)

        # Add abnormal pairs
        abnormal_mask = is_abnormal
        abnormal_chunk_pairs = set(zip(pids[abnormal_mask], lab_ids[abnormal_mask]))
        abnormal_pairs.update(abnormal_chunk_pairs)

    logger.info(f"Finished processing {total_chunks} chunks, {total_rows:,} total rows")
    logger.info(f"Found {len(all_pairs):,} unique (patient, lab) pairs")
    logger.info(f"Found {len(abnormal_pairs):,} abnormal (patient, lab) pairs")

    return abnormal_pairs, all_pairs, patient_labels


def compute_ntk_scores(abnormal_pairs, all_pairs, patient_labels, logger):
    """
    Compute NTK-style scores for each lab.

    ntk_score_lab_j = mean over all patients of |(0.5 - y_i) * x_{i,j}|

    Where:
    - y_i is the label for patient i (0 or 1)
    - x_{i,j} is 1 if patient i has abnormal result for lab j, 0 otherwise
    """
    logger.info("Computing NTK scores at patient level...")

    # Get all unique patients and labs
    all_patients = list(patient_labels.keys())
    all_labs = list(set(lab_id for _, lab_id in all_pairs))

    n_patients = len(all_patients)
    n_labs = len(all_labs)

    logger.info(f"Number of patients with labels: {n_patients:,}")
    logger.info(f"Number of unique labs: {n_labs:,}")

    # Precompute patient labels array and counts
    patient_to_idx = {pid: i for i, pid in enumerate(all_patients)}
    labels_array = np.array([patient_labels[pid] for pid in all_patients])
    n_pos_total = int(np.sum(labels_array == 1))
    n_neg_total = int(np.sum(labels_array == 0))

    # Precompute gradient weights: |0.5 - y_i|
    # For y=1: |0.5 - 1| = 0.5
    # For y=0: |0.5 - 0| = 0.5
    # So gradient weight is always 0.5 when x_ij = 1
    grad_weight = 0.5

    # Build lookup: lab_id -> set of patient indices with abnormal
    logger.info("Building lab -> patient abnormal index...")
    lab_to_abnormal_patients = {}
    for pid, lab_id in tqdm(abnormal_pairs, desc="Indexing abnormal pairs"):
        if pid in patient_to_idx:
            if lab_id not in lab_to_abnormal_patients:
                lab_to_abnormal_patients[lab_id] = set()
            lab_to_abnormal_patients[lab_id].add(patient_to_idx[pid])

    # Compute scores for each lab
    logger.info("Computing NTK scores for each lab...")
    lab_results = []

    for lab_id in tqdm(all_labs, desc="Computing NTK scores"):
        abnormal_patient_indices = lab_to_abnormal_patients.get(lab_id, set())
        n_patients_with_abnormal = len(abnormal_patient_indices)

        if n_patients_with_abnormal == 0:
            # No abnormal results for this lab
            lab_results.append({
                'DETAIL_LAB_PROCEDURE_ID': lab_id,
                'ntk_score_lab': 0.0,
                'n_patients_with_abnormal': 0,
                'abnormal_rate_patients_pos': 0.0,
                'abnormal_rate_patients_neg': 0.0
            })
            continue

        # Create boolean mask for patients with abnormal
        abnormal_mask = np.zeros(n_patients, dtype=bool)
        for idx in abnormal_patient_indices:
            abnormal_mask[idx] = True

        # NTK score: mean of |0.5 - y_i| * x_ij = 0.5 * mean(x_ij) = 0.5 * (n_abnormal / n_patients)
        # Since |0.5 - y| = 0.5 for both y=0 and y=1
        ntk_score = grad_weight * n_patients_with_abnormal / n_patients

        # Compute abnormal rates by label
        abnormal_labels = labels_array[abnormal_mask]
        n_pos_with_abnormal = int(np.sum(abnormal_labels == 1))
        n_neg_with_abnormal = int(np.sum(abnormal_labels == 0))

        abnormal_rate_pos = n_pos_with_abnormal / n_pos_total if n_pos_total > 0 else 0.0
        abnormal_rate_neg = n_neg_with_abnormal / n_neg_total if n_neg_total > 0 else 0.0

        lab_results.append({
            'DETAIL_LAB_PROCEDURE_ID': lab_id,
            'ntk_score_lab': ntk_score,
            'n_patients_with_abnormal': n_patients_with_abnormal,
            'abnormal_rate_patients_pos': abnormal_rate_pos,
            'abnormal_rate_patients_neg': abnormal_rate_neg
        })

    results_df = pd.DataFrame(lab_results)
    results_df = results_df.sort_values('ntk_score_lab', ascending=False).reset_index(drop=True)

    return results_df


def merge_lab_names(results_df, logger):
    """
    Merge lab procedure names from the dictionary table.
    """
    logger.info(f"Loading lab dictionary from: {INPUT_LAB_DICT}")

    lab_dict = pd.read_csv(INPUT_LAB_DICT, usecols=['LAB_PROCEDURE_ID', 'LAB_PROCEDURE_NAME'])

    # Merge (dictionary uses LAB_PROCEDURE_ID, our data uses DETAIL_LAB_PROCEDURE_ID)
    results_df = results_df.merge(
        lab_dict,
        left_on='DETAIL_LAB_PROCEDURE_ID',
        right_on='LAB_PROCEDURE_ID',
        how='left'
    ).drop(columns=['LAB_PROCEDURE_ID'])

    # Reorder columns
    cols = ['DETAIL_LAB_PROCEDURE_ID', 'LAB_PROCEDURE_NAME', 'ntk_score_lab',
            'n_patients_with_abnormal', 'abnormal_rate_patients_pos', 'abnormal_rate_patients_neg']
    results_df = results_df[cols]

    return results_df


def main():
    """Main execution function."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("NTK-style Lab Feature Selection - Starting")
    logger.info("=" * 60)

    # Step 1: Process lab data in chunks
    abnormal_pairs, all_pairs, patient_labels = process_lab_data_in_chunks(logger)

    # Log patient/label statistics
    n_patients = len(patient_labels)
    n_pos = sum(1 for v in patient_labels.values() if v == 1)
    n_neg = sum(1 for v in patient_labels.values() if v == 0)
    logger.info(f"Total patients: {n_patients:,} (OUD: {n_pos:,}, non-OUD: {n_neg:,})")

    # Step 2: Compute NTK scores
    results_df = compute_ntk_scores(abnormal_pairs, all_pairs, patient_labels, logger)

    # Checkpoint: save results before merge (in case merge fails)
    checkpoint_file = os.path.join(OUTPUT_DIR, "lab_ntk_patient_level_no_names.csv")
    results_df.to_csv(checkpoint_file, index=False)
    logger.info(f"Checkpoint saved to: {checkpoint_file}")

    # Step 3: Merge lab names
    results_df = merge_lab_names(results_df, logger)

    # Step 4: Save results
    output_file = os.path.join(OUTPUT_DIR, "lab_ntk_patient_level.csv")
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved NTK results to: {output_file}")

    # Log summary statistics
    logger.info("=" * 60)
    logger.info("Summary Statistics")
    logger.info("=" * 60)
    logger.info(f"Number of patients: {n_patients:,}")
    logger.info(f"Number of labs: {len(results_df):,}")
    logger.info(f"Number of (patient, lab) abnormal pairs: {len(abnormal_pairs):,}")

    # Log top 20 labs
    logger.info("=" * 60)
    logger.info("Top 20 Labs by NTK Score")
    logger.info("=" * 60)
    top20 = results_df.head(20)
    for idx, row in top20.iterrows():
        logger.info(
            f"  {idx+1:2d}. {row['DETAIL_LAB_PROCEDURE_ID']:>10} | "
            f"NTK={row['ntk_score_lab']:.6f} | "
            f"n_abnormal={row['n_patients_with_abnormal']:>6} | "
            f"{row['LAB_PROCEDURE_NAME']}"
        )

    logger.info("=" * 60)
    logger.info("NTK-style Lab Feature Selection - Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
