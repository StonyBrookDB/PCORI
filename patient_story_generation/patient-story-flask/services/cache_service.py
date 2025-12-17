import json
import logging
from typing import Dict, List, Optional, Tuple

import mysql.connector

from config import Config

# In-memory cache
_cache_by_fingerprint: Dict[Tuple[str, str], dict] = {}
_latest_by_patient: Dict[str, dict] = {}

cfg = Config()
logger = logging.getLogger(__name__)


def _db_conn():
    if not cfg.use_db_cache:
        return None
    try:
        return mysql.connector.connect(
            host=cfg.db_host,
            port=cfg.db_port,
            user=cfg.db_user,
            password=cfg.db_password,
            database=cfg.db_name,
        )
    except Exception:  # noqa: BLE001
        logger.warning("DB connection failed", exc_info=True)
        return None


def get_by_fingerprint(patient_id: str, fp: str) -> Optional[dict]:
    cached = _cache_by_fingerprint.get((patient_id, fp))
    if cached:
        return cached
    conn = _db_conn()
    if not conn:
        return None
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            f"""
            SELECT patient_id, fingerprint, model_name, prompt_version,
                   clinician_summary, patient_story, disclaimer, generated_at, token_usage
            FROM {cfg.story_cache_table}
            WHERE patient_id=%s AND fingerprint=%s
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (patient_id, fp),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            _cache_by_fingerprint[(patient_id, fp)] = row
            _latest_by_patient[patient_id] = row
            return row
    except Exception:  # noqa: BLE001
        logger.exception("get_by_fingerprint failed for patient_id=%s", patient_id)
    return None


def get_latest(patient_id: str) -> Optional[dict]:
    cached = _latest_by_patient.get(patient_id)
    if cached:
        return cached
    conn = _db_conn()
    if not conn:
        return None
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            f"""
            SELECT patient_id, fingerprint, model_name, prompt_version,
                   clinician_summary, patient_story, disclaimer, generated_at, token_usage
            FROM {cfg.story_cache_table}
            WHERE patient_id=%s
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (patient_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            _cache_by_fingerprint[(patient_id, row["fingerprint"])] = row
            _latest_by_patient[patient_id] = row
            return row
    except Exception:  # noqa: BLE001
        logger.exception("get_latest failed for patient_id=%s", patient_id)
    return None


def get_history(patient_id: str, limit: int = 50) -> List[dict]:
    """
    Return all story versions for a patient_id (newest first).
    """
    # Fast path for in-memory only
    if not cfg.use_db_cache:
        rows = [v for (pid, _fp), v in _cache_by_fingerprint.items() if pid == patient_id]
        rows.sort(key=lambda r: r.get("generated_at", ""), reverse=True)
        return rows[: max(1, limit)]

    conn = _db_conn()
    if not conn:
        return []
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            f"""
            SELECT id, patient_id, fingerprint, model_name, prompt_version,
                   clinician_summary, patient_story, disclaimer, generated_at, token_usage
            FROM {cfg.story_cache_table}
            WHERE patient_id=%s
            ORDER BY generated_at DESC, id DESC
            LIMIT %s
            """,
            (patient_id, max(1, limit)),
        )
        rows = cur.fetchall() or []
        cur.close()
        conn.close()
        return rows
    except Exception:  # noqa: BLE001
        logger.exception("get_history failed for patient_id=%s", patient_id)
        return []


def save(record: dict) -> None:
    _cache_by_fingerprint[(record["patient_id"], record["fingerprint"])] = record
    _latest_by_patient[record["patient_id"]] = record

    conn = _db_conn()
    if not conn:
        return
    try:
        cur = conn.cursor()

        # History mode: insert a new row when fingerprint changes.
        # If the same fingerprint already exists for this patient_id, skip insert.
        cur.execute(
            f"SELECT id FROM {cfg.story_cache_table} WHERE patient_id=%s AND fingerprint=%s LIMIT 1",
            (record["patient_id"], record["fingerprint"]),
        )
        exists = cur.fetchone()
        if not exists:
            cur.execute(
                f"""
                INSERT INTO {cfg.story_cache_table} (
                    patient_id, fingerprint, model_name, prompt_version,
                    clinician_summary, patient_story, disclaimer, generated_at, token_usage
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    record["patient_id"],
                    record["fingerprint"],
                    record["model_name"],
                    record["prompt_version"],
                    record["clinician_summary"],
                    record["patient_story"],
                    record["disclaimer"],
                    record["generated_at"],
                    json.dumps(record.get("token_usage", {})),
                ),
            )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:  # noqa: BLE001
        logger.exception(
            "save failed for patient_id=%s fingerprint=%s",
            record.get("patient_id"),
            record.get("fingerprint"),
        )
