"""Data service to fetch patient data from MySQL by patient_id."""

from __future__ import annotations

import json
import os
from typing import Optional

try:
    import mysql.connector  # type: ignore
except ImportError:  # pragma: no cover
    mysql = None  # type: ignore

USE_DB = os.getenv("USE_DB_CACHE", "false").lower() == "true"
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "patient_story"),
}
PATIENT_TABLE = os.getenv("PATIENT_TABLE", "patient_data")


def _get_conn():
    if not USE_DB or mysql is None:
        return None
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as exc:  # noqa: BLE001
        print(f"[DB] Connection failed in data_service: {exc}")
        return None


def fetch_patient_data(patient_id: str) -> Optional[dict]:
    """Fetch patient payload JSON from patient_data table."""
    conn = _get_conn()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT payload FROM {PATIENT_TABLE} WHERE patient_id=%s LIMIT 1", (patient_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if not row:
            return None
        payload = row[0]
        if isinstance(payload, str):
            return json.loads(payload)
        if isinstance(payload, bytes):
            return json.loads(payload.decode())
        if isinstance(payload, dict):
            return payload
    except Exception as exc:  # noqa: BLE001
        print(f"[DB] fetch_patient_data failed: {exc}")
    return None
