"""
Build a patient-level table from diagnosis.csv.

Output: data/patient_diagnosis.csv with 3 columns
  - PATIENT_ID  : int
  - ICD10_LIST  : space-separated string of ICD-10 codes in chronological order
                  (read back into a list with `s.split()`)
  - LABEL       : 1 if the patient has any row with LABEL=1, else 0

Chronological order
-------------------
diagnosis.csv has no timestamp column. ENCOUNTER_ID is used as the time proxy
(common in EHR systems where encounter IDs are assigned sequentially in time).
Codes are sorted ascending by ENCOUNTER_ID per patient; codes belonging to the
same encounter keep their original CSV row order.

Run with:
  python src/build_patient_table.py
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_CSV = DATA_DIR / "diagnosis.csv"
OUTPUT_CSV = DATA_DIR / "patient_diagnosis.csv"
CHUNK_SIZE = 500_000


def build_table():
    # patient_id -> list[(encounter_id, icd10_code, row_order_within_chunk)]
    # row_order keeps within-encounter codes stable when sorting
    patient_rows: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
    oud_patients: set[int] = set()

    global_row_counter = 0
    t0 = time.time()

    reader = pd.read_csv(
        INPUT_CSV,
        dtype={
            "PATIENT_ID": np.int64,
            "ENCOUNTER_ID": np.int64,
            "ICD10_3Digits": str,
            "LABEL": np.int8,
        },
        chunksize=CHUNK_SIZE,
    )

    for chunk_num, chunk in enumerate(reader, start=1):
        # Record OUD-positive patients
        pos_pids = chunk.loc[chunk["LABEL"] == 1, "PATIENT_ID"].unique()
        oud_patients.update(int(p) for p in pos_pids)

        # Accumulate (encounter_id, code, row_order) per patient
        pids = chunk["PATIENT_ID"].to_numpy()
        eids = chunk["ENCOUNTER_ID"].to_numpy()
        codes = chunk["ICD10_3Digits"].to_numpy()
        for i in range(len(chunk)):
            patient_rows[int(pids[i])].append(
                (int(eids[i]), str(codes[i]), global_row_counter + i)
            )
        global_row_counter += len(chunk)

        if chunk_num % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  chunk {chunk_num:4d} | {elapsed:6.0f}s | "
                f"patients seen: {len(patient_rows):,}",
                flush=True,
            )

    print(
        f"\nFinished reading. Patients: {len(patient_rows):,}  "
        f"OUD-positive: {len(oud_patients):,} "
        f"({100 * len(oud_patients) / max(len(patient_rows), 1):.2f}%)",
        flush=True,
    )

    # Write output CSV
    print(f"\nWriting {OUTPUT_CSV} ...", flush=True)
    t1 = time.time()
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PATIENT_ID", "ICD10_LIST", "LABEL"])

        # Sort patient IDs for a stable output ordering
        for pid in sorted(patient_rows):
            triples = patient_rows[pid]
            # Sort by encounter_id then by original row order (stable tie-break)
            triples.sort(key=lambda t: (t[0], t[2]))
            codes_str = " ".join(t[1] for t in triples)
            label = 1 if pid in oud_patients else 0
            writer.writerow([pid, codes_str, label])

    print(
        f"Wrote {OUTPUT_CSV}  ({time.time() - t1:.0f}s write time, "
        f"{time.time() - t0:.0f}s total)",
        flush=True,
    )


if __name__ == "__main__":
    build_table()
