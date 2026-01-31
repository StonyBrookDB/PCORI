#!/usr/bin/env python3
"""
Compute lab abnormality summary tables (optimized version).

This script processes lab test data to compute abnormality rates at three levels:
- Record-level (row-level)
- Encounter-level
- Patient-level

Each level is also stratified by label (POS/NEG).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Input/Output paths
LAB_RAW_PATH = "/home/zihan/pcori_project/final_data/processed_data/encounter5/lab_test_with_patient_label.csv"
LAB_COVERAGE_PATH = "/home/zihan/pcori_project/final_data/processed_data/encounter5/lab_coverage_summary.csv"
OUTPUT_DIR = Path("/home/zihan/pcori_project/final_data/feature_selection/labs")

# Indicator values that indicate abnormality (case-sensitive)
ABNORMAL_INDICATORS = frozenset({"Low", "High", "Abnormal", "Critical", "Outside limits"})

# Chunk size for processing large file
CHUNK_SIZE = 10_000_000  # Larger chunks = fewer iterations


def compute_abnormality_flags(df):
    """Compute abnormality flags using vectorized operations."""
    # Convert to numeric, coercing errors to NaN
    numeric_result = pd.to_numeric(df['NUMERIC_RESULT'], errors='coerce')
    range_low = pd.to_numeric(df['NORMAL_RANGE_LOW'], errors='coerce')
    range_high = pd.to_numeric(df['NORMAL_RANGE_HIGH'], errors='coerce')

    # Primary rule: numeric-based (all three must be non-missing)
    numeric_valid = numeric_result.notna() & range_low.notna() & range_high.notna()
    abnormal_numeric = numeric_valid & ((numeric_result < range_low) | (numeric_result > range_high))

    # Secondary rule: indicator-based fallback
    indicator = df['RESULT_INDICATOR_DESC']
    indicator_valid = indicator.notna()
    indicator_abnormal = indicator.isin(ABNORMAL_INDICATORS)
    abnormal_indicator = ~numeric_valid & indicator_valid & indicator_abnormal

    # Valid for assessment and final abnormal flag
    valid = (numeric_valid | indicator_valid).values
    abnormal = (abnormal_numeric | abnormal_indicator).values

    return valid, abnormal


def process_file():
    """Process the lab file and return aggregated DataFrames for each level."""

    # Accumulators for record-level (list of partial aggregations)
    record_parts = []

    # Accumulators for encounter/patient level (list of DataFrames to concat)
    encounter_parts = []
    patient_parts = []

    total_rows = 0
    label_mismatch_enc = 0
    label_mismatch_pat = 0

    print(f"Processing {LAB_RAW_PATH} in chunks of {CHUNK_SIZE:,} rows...")

    for chunk_idx, chunk in enumerate(pd.read_csv(
        LAB_RAW_PATH,
        chunksize=CHUNK_SIZE,
        usecols=['ENCOUNTER_ID', 'DETAIL_LAB_PROCEDURE_ID', 'NUMERIC_RESULT',
                 'NORMAL_RANGE_LOW', 'NORMAL_RANGE_HIGH', 'RESULT_INDICATOR_DESC',
                 'PATIENT_ID', 'LABEL'],
        dtype={'ENCOUNTER_ID': 'int64', 'DETAIL_LAB_PROCEDURE_ID': 'int32',
               'PATIENT_ID': 'int64', 'LABEL': 'int8'}
    )):
        total_rows += len(chunk)
        print(f"  Chunk {chunk_idx + 1}: {total_rows:,} rows processed...")

        # Compute flags
        valid, abnormal = compute_abnormality_flags(chunk)

        # Filter to valid records
        valid_mask = valid
        if not valid_mask.any():
            continue

        valid_df = chunk.loc[valid_mask, ['DETAIL_LAB_PROCEDURE_ID', 'ENCOUNTER_ID',
                                           'PATIENT_ID', 'LABEL']].copy()
        valid_df['abnormal'] = abnormal[valid_mask]

        # --- Record-level aggregation ---
        rec_agg = valid_df.groupby(['DETAIL_LAB_PROCEDURE_ID', 'LABEL']).agg(
            n_valid=('abnormal', 'size'),
            n_abnormal=('abnormal', 'sum')
        ).reset_index()
        record_parts.append(rec_agg)

        # --- Encounter-level: get (lab, encounter) -> (any_abnormal, label) ---
        enc_agg = valid_df.groupby(['DETAIL_LAB_PROCEDURE_ID', 'ENCOUNTER_ID']).agg(
            abnormal=('abnormal', 'any'),
            label=('LABEL', 'first'),
            label_nunique=('LABEL', 'nunique')
        ).reset_index()
        label_mismatch_enc += (enc_agg['label_nunique'] > 1).sum()
        encounter_parts.append(enc_agg[['DETAIL_LAB_PROCEDURE_ID', 'ENCOUNTER_ID', 'abnormal', 'label']])

        # --- Patient-level: get (lab, patient) -> (any_abnormal, label) ---
        pat_agg = valid_df.groupby(['DETAIL_LAB_PROCEDURE_ID', 'PATIENT_ID']).agg(
            abnormal=('abnormal', 'any'),
            label=('LABEL', 'first'),
            label_nunique=('LABEL', 'nunique')
        ).reset_index()
        label_mismatch_pat += (pat_agg['label_nunique'] > 1).sum()
        patient_parts.append(pat_agg[['DETAIL_LAB_PROCEDURE_ID', 'PATIENT_ID', 'abnormal', 'label']])

    print(f"\nTotal rows processed: {total_rows:,}")
    if label_mismatch_enc > 0:
        print(f"WARNING: {label_mismatch_enc:,} encounter groups had label mismatches (using first)")
    if label_mismatch_pat > 0:
        print(f"WARNING: {label_mismatch_pat:,} patient groups had label mismatches (using first)")

    # Combine all parts
    print("\nCombining record-level aggregations...")
    record_df = pd.concat(record_parts, ignore_index=True)
    record_df = record_df.groupby(['DETAIL_LAB_PROCEDURE_ID', 'LABEL']).agg(
        n_valid=('n_valid', 'sum'),
        n_abnormal=('n_abnormal', 'sum')
    ).reset_index()

    print("Combining encounter-level aggregations...")
    encounter_df = pd.concat(encounter_parts, ignore_index=True)
    # Deduplicate: same (lab, encounter) may appear in multiple chunks
    # Keep abnormal=True if any chunk had it (OR logic)
    encounter_df = encounter_df.groupby(['DETAIL_LAB_PROCEDURE_ID', 'ENCOUNTER_ID']).agg(
        abnormal=('abnormal', 'any'),
        label=('label', 'first')
    ).reset_index()

    print("Combining patient-level aggregations...")
    patient_df = pd.concat(patient_parts, ignore_index=True)
    # Deduplicate: same (lab, patient) may appear in multiple chunks
    patient_df = patient_df.groupby(['DETAIL_LAB_PROCEDURE_ID', 'PATIENT_ID']).agg(
        abnormal=('abnormal', 'any'),
        label=('label', 'first')
    ).reset_index()

    return record_df, encounter_df, patient_df


def compute_record_level_summary(record_df):
    """Compute record-level summary from aggregated data."""
    # Pivot to get pos/neg columns
    overall = record_df.groupby('DETAIL_LAB_PROCEDURE_ID').agg(
        n_valid_records=('n_valid', 'sum'),
        n_abnormal_records=('n_abnormal', 'sum')
    )

    pos = record_df[record_df['LABEL'] == 1].groupby('DETAIL_LAB_PROCEDURE_ID').agg(
        n_valid_records_pos=('n_valid', 'sum'),
        n_abnormal_records_pos=('n_abnormal', 'sum')
    )

    neg = record_df[record_df['LABEL'] == 0].groupby('DETAIL_LAB_PROCEDURE_ID').agg(
        n_valid_records_neg=('n_valid', 'sum'),
        n_abnormal_records_neg=('n_abnormal', 'sum')
    )

    result = overall.join(pos, how='left').join(neg, how='left').fillna(0)
    result['abnormal_rate_records'] = result['n_abnormal_records'] / result['n_valid_records']
    result['abnormal_rate_records_pos'] = result['n_abnormal_records_pos'] / result['n_valid_records_pos']
    result['abnormal_rate_records_neg'] = result['n_abnormal_records_neg'] / result['n_valid_records_neg']

    return result.reset_index()


def compute_entity_level_summary(entity_df, level_name):
    """Compute encounter or patient level summary."""
    entity_df = entity_df.copy()
    entity_df['abnormal_int'] = entity_df['abnormal'].astype(int)

    # Overall
    overall = entity_df.groupby('DETAIL_LAB_PROCEDURE_ID').agg(
        n_valid=('abnormal', 'size'),
        n_abnormal=('abnormal_int', 'sum')
    )
    overall.columns = [f'n_valid_{level_name}s', f'n_abnormal_{level_name}s']

    # By label
    pos = entity_df[entity_df['label'] == 1].groupby('DETAIL_LAB_PROCEDURE_ID').agg(
        n_valid=('abnormal', 'size'),
        n_abnormal=('abnormal_int', 'sum')
    )
    pos.columns = [f'n_valid_{level_name}s_pos', f'n_abnormal_{level_name}s_pos']

    neg = entity_df[entity_df['label'] == 0].groupby('DETAIL_LAB_PROCEDURE_ID').agg(
        n_valid=('abnormal', 'size'),
        n_abnormal=('abnormal_int', 'sum')
    )
    neg.columns = [f'n_valid_{level_name}s_neg', f'n_abnormal_{level_name}s_neg']

    result = overall.join(pos, how='left').join(neg, how='left').fillna(0)

    # Compute rates
    result[f'abnormal_rate_{level_name}s'] = (
        result[f'n_abnormal_{level_name}s'] / result[f'n_valid_{level_name}s']
    )
    result[f'abnormal_rate_{level_name}s_pos'] = (
        result[f'n_abnormal_{level_name}s_pos'] / result[f'n_valid_{level_name}s_pos']
    )
    result[f'abnormal_rate_{level_name}s_neg'] = (
        result[f'n_abnormal_{level_name}s_neg'] / result[f'n_valid_{level_name}s_neg']
    )

    return result.reset_index()


def main():
    print("=" * 60)
    print("Lab Abnormality Summary Computation (Optimized)")
    print("=" * 60)

    # Step 1: Process file
    record_df, encounter_df, patient_df = process_file()

    print(f"\nUnique labs: {record_df['DETAIL_LAB_PROCEDURE_ID'].nunique():,}")
    print(f"Unique (lab, encounter) pairs: {len(encounter_df):,}")
    print(f"Unique (lab, patient) pairs: {len(patient_df):,}")

    # Step 2: Compute summaries
    print("\nComputing record-level summary...")
    record_summary = compute_record_level_summary(record_df)

    print("Computing encounter-level summary...")
    encounter_summary = compute_entity_level_summary(encounter_df, 'encounter')

    print("Computing patient-level summary...")
    patient_summary = compute_entity_level_summary(patient_df, 'patient')

    # Step 3: Merge all summaries
    print("Merging summaries...")
    summary_df = record_summary.merge(
        encounter_summary, on='DETAIL_LAB_PROCEDURE_ID', how='outer'
    ).merge(
        patient_summary, on='DETAIL_LAB_PROCEDURE_ID', how='outer'
    )

    # Step 4: Merge with coverage stats
    print(f"Merging with coverage data...")
    coverage_df = pd.read_csv(LAB_COVERAGE_PATH)
    summary_df = summary_df.merge(coverage_df, on='DETAIL_LAB_PROCEDURE_ID', how='left')

    # Reorder columns
    col_order = [
        'DETAIL_LAB_PROCEDURE_ID',
        'n_valid_records', 'n_abnormal_records', 'abnormal_rate_records',
        'n_valid_records_pos', 'n_abnormal_records_pos', 'abnormal_rate_records_pos',
        'n_valid_records_neg', 'n_abnormal_records_neg', 'abnormal_rate_records_neg',
        'n_valid_encounters', 'n_abnormal_encounters', 'abnormal_rate_encounters',
        'n_valid_encounters_pos', 'n_abnormal_encounters_pos', 'abnormal_rate_encounters_pos',
        'n_valid_encounters_neg', 'n_abnormal_encounters_neg', 'abnormal_rate_encounters_neg',
        'n_valid_patients', 'n_abnormal_patients', 'abnormal_rate_patients',
        'n_valid_patients_pos', 'n_abnormal_patients_pos', 'abnormal_rate_patients_pos',
        'n_valid_patients_neg', 'n_abnormal_patients_neg', 'abnormal_rate_patients_neg',
        'n_patients_with_lab', 'n_encounters_with_lab'
    ]
    summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

    # Step 5: Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "lab_abnormal_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"Total labs: {len(summary_df):,}")

    # Print statistics
    print("\n" + "=" * 60)
    print("Summary Statistics (median abnormal rates):")
    print("=" * 60)
    for col in ['abnormal_rate_records', 'abnormal_rate_encounters', 'abnormal_rate_patients']:
        if col in summary_df.columns:
            print(f"  {col}: {summary_df[col].median():.4f}")

    print(f"\nColumns: {list(summary_df.columns)}")
    print("\nFirst 5 rows:")
    print(summary_df.head().to_string())

    return summary_df


if __name__ == "__main__":
    main()
