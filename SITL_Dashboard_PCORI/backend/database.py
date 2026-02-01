"""
Database module for PCORI SITL Dashboard
Handles SQLite connection and patient queries
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import DB_PATH, MIN_COHORT_SIZE, MIN_EVENT_COUNT


def get_db_path() -> str:
    """Get database path, creating directory if needed."""
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_schema():
    """Initialize database schema."""
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT,
                age INTEGER NOT NULL,
                cancer_status INTEGER DEFAULT 0,
                prior_opioid_days INTEGER NOT NULL,
                mental_health_score INTEGER NOT NULL,
                benzo_use INTEGER DEFAULT 0,
                er_visits INTEGER NOT NULL,
                outcome INTEGER NOT NULL,
                split TEXT DEFAULT 'train',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indexes for fast filtering
        conn.execute("CREATE INDEX IF NOT EXISTS idx_age ON patients(age)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cancer ON patients(cancer_status)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_opioid ON patients(prior_opioid_days)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_outcome ON patients(outcome)")

        conn.commit()


def get_patient_count() -> int:
    """Get total patient count."""
    with get_db() as conn:
        return conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]


def filter_patients(
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    cancer_status: Optional[bool] = None,
    opioid_threshold: Optional[int] = None,
    mental_health_min: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Filter patients based on criteria.

    Returns:
        Dictionary with patient_ids, counts, statistics, and warnings
    """
    conditions = []
    params = []

    if age_min is not None:
        conditions.append("age >= ?")
        params.append(age_min)

    if age_max is not None:
        conditions.append("age <= ?")
        params.append(age_max)

    if cancer_status is not None:
        conditions.append("cancer_status = ?")
        params.append(1 if cancer_status else 0)

    if opioid_threshold is not None:
        conditions.append("prior_opioid_days >= ?")
        params.append(opioid_threshold)

    if mental_health_min is not None:
        conditions.append("mental_health_score >= ?")
        params.append(mental_health_min)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    with get_db() as conn:
        # Get filtered patients
        query = f"SELECT * FROM patients WHERE {where_clause}"
        rows = conn.execute(query, params).fetchall()

        if not rows:
            return {
                "patient_ids": [],
                "total_count": 0,
                "event_count": 0,
                "event_rate": 0.0,
                "train_count": 0,
                "val_count": 0,
                "statistics": {},
                "warnings": ["No patients match the filter criteria"],
            }

        # Extract data
        patient_ids = [r["patient_id"] for r in rows]
        outcomes = [r["outcome"] for r in rows]
        ages = [r["age"] for r in rows]
        splits = [r["split"] for r in rows]

        total_count = len(patient_ids)
        event_count = sum(outcomes)
        non_event_count = total_count - event_count
        event_rate = event_count / total_count if total_count > 0 else 0

        train_count = sum(1 for s in splits if s == "train")
        val_count = total_count - train_count

        # Generate warnings
        warnings = []
        if total_count < MIN_COHORT_SIZE:
            warnings.append(
                f"Cohort size ({total_count}) below recommended minimum ({MIN_COHORT_SIZE})"
            )
        if event_count < MIN_EVENT_COUNT:
            warnings.append(
                f"Event count ({event_count}) below recommended minimum ({MIN_EVENT_COUNT})"
            )
        if non_event_count < MIN_EVENT_COUNT:
            warnings.append(
                f"Non-event count ({non_event_count}) below recommended minimum ({MIN_EVENT_COUNT})"
            )

        return {
            "patient_ids": patient_ids,
            "total_count": total_count,
            "event_count": event_count,
            "event_rate": round(event_rate, 4),
            "train_count": train_count,
            "val_count": val_count,
            "statistics": {
                "age_mean": round(float(np.mean(ages)), 2),
                "age_std": round(float(np.std(ages)), 2),
                "age_min": int(min(ages)),
                "age_max": int(max(ages)),
            },
            "warnings": warnings,
        }


def get_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    """Get single patient by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()

        if row:
            return dict(row)
        return None


def get_patients_by_ids(patient_ids: List[str]) -> List[Dict[str, Any]]:
    """Get multiple patients by IDs."""
    if not patient_ids:
        return []

    placeholders = ",".join("?" * len(patient_ids))
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM patients WHERE patient_id IN ({placeholders})", patient_ids
        ).fetchall()
        return [dict(r) for r in rows]


def get_training_data(patient_ids: List[str], features: List[str]) -> tuple:
    """
    Get training data for specified patients and features.

    Returns:
        (X_train, y_train, X_val, y_val, feature_names)
    """
    patients = get_patients_by_ids(patient_ids)

    train_X, train_y = [], []
    val_X, val_y = [], []

    for p in patients:
        x = [p[f] for f in features]
        y = p["outcome"]

        if p["split"] == "train":
            train_X.append(x)
            train_y.append(y)
        else:
            val_X.append(x)
            val_y.append(y)

    return (
        np.array(train_X, dtype=np.float32),
        np.array(train_y, dtype=np.float32),
        np.array(val_X, dtype=np.float32),
        np.array(val_y, dtype=np.float32),
        features,
    )


def insert_patient(patient: Dict[str, Any]):
    """Insert a single patient."""
    with get_db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO patients
            (patient_id, name, age, cancer_status, prior_opioid_days,
             mental_health_score, benzo_use, er_visits, outcome, split)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                patient["patient_id"],
                patient.get("name", ""),
                patient["age"],
                patient.get("cancer_status", 0),
                patient["prior_opioid_days"],
                patient["mental_health_score"],
                patient.get("benzo_use", 0),
                patient["er_visits"],
                patient["outcome"],
                patient.get("split", "train"),
            ),
        )
        conn.commit()


def insert_patients_bulk(patients: List[Dict[str, Any]]):
    """Bulk insert patients."""
    with get_db() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO patients
            (patient_id, name, age, cancer_status, prior_opioid_days,
             mental_health_score, benzo_use, er_visits, outcome, split)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                (
                    p["patient_id"],
                    p.get("name", ""),
                    p["age"],
                    p.get("cancer_status", 0),
                    p["prior_opioid_days"],
                    p["mental_health_score"],
                    p.get("benzo_use", 0),
                    p["er_visits"],
                    p["outcome"],
                    p.get("split", "train"),
                )
                for p in patients
            ],
        )
        conn.commit()


def clear_patients():
    """Clear all patients from database."""
    with get_db() as conn:
        conn.execute("DELETE FROM patients")
        conn.commit()
