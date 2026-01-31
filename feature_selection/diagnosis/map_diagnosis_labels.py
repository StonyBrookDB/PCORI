#!/usr/bin/env python3
"""
Map patient labels to diagnosis data.
1. Read diagnosis_icd10.csv (PATIENT_ID, ENCOUNTER_ID, ICD10_3Digits)
2. Read patient_oud_labels.csv (PATIENT_ID, LABEL)
3. Join to add LABEL column
4. Write to feature_selection/diagnosis.csv
"""

import pandas as pd
import os

# File paths
DIAGNOSIS_FILE = "/home/zihan/pcori_project/final_data/processed_data/encounter5/diagnosis_icd10.csv"
LABELS_FILE = "/home/zihan/pcori_project/final_data/processed_data/patient_oud_labels.csv"
OUTPUT_FILE = "/home/zihan/pcori_project/final_data/feature_selection/diagnosis.csv"

CHUNK_SIZE = 1_000_000


def main():
    print("=" * 50)
    print("Mapping Labels to Diagnosis Data")
    print("=" * 50)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load labels (small table, fits in memory)
    print(f"\nLoading labels from {LABELS_FILE}...")
    labels_df = pd.read_csv(
        LABELS_FILE,
        usecols=["PATIENT_ID", "LABEL"],
        dtype={"PATIENT_ID": str, "LABEL": int}
    )
    print(f"  Loaded {len(labels_df):,} patient labels")

    # Process diagnosis file in chunks
    print(f"\nProcessing diagnosis file: {DIAGNOSIS_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    total_rows = 0
    first_chunk = True

    reader = pd.read_csv(
        DIAGNOSIS_FILE,
        usecols=["PATIENT_ID", "ENCOUNTER_ID", "ICD10_3Digits"],
        dtype=str,
        chunksize=CHUNK_SIZE
    )

    for chunk_num, chunk in enumerate(reader):
        # Left join with labels
        chunk = chunk.merge(labels_df, on="PATIENT_ID", how="left")

        # Write to output
        chunk.to_csv(
            OUTPUT_FILE,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False
        )

        total_rows += len(chunk)
        first_chunk = False

        print(f"  Chunk {chunk_num + 1}: {total_rows:,} rows processed")

    print(f"\nCompleted! Total rows: {total_rows:,}")
    print(f"Output written to: {OUTPUT_FILE}")

    # Show sample
    print("\n--- Sample output (first 5 rows) ---")
    sample = pd.read_csv(OUTPUT_FILE, nrows=5)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
