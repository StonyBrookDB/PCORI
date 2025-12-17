"""Cache layer for patient stories.

Supports in-memory caching and optional MySQL persistence when USE_DB_CACHE=true.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

from .models import PatientStoryResponse

try:
    import mysql.connector  # type: ignore
except ImportError:  # pragma: no cover
    mysql = None  # type: ignore

# Keyed by (patient_id, fingerprint) to preserve history per patient.
_cache_by_fingerprint: Dict[Tuple[str, str], PatientStoryResponse] = {}
# Latest story per patient for quick GET /story/{patient_id}.
_latest_by_patient: Dict[str, PatientStoryResponse] = {}

USE_DB_CACHE = os.getenv("USE_DB_CACHE", "false").lower() == "true"
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "patient_story"),
}


def _get_conn():
    if not USE_DB_CACHE or mysql is None:
        return None
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as exc:  # noqa: BLE001
        print(f"[DB] Connection failed, using in-memory cache: {exc}")
        return None


def _row_to_story(row) -> PatientStoryResponse:
    (
        patient_id,
        fingerprint,
        model_name,
        prompt_version,
        clinician_summary,
        patient_story,
        disclaimer,
        generated_at,
    ) = row
    return PatientStoryResponse(
        patient_id=patient_id,
        fingerprint=fingerprint,
        model_name=model_name,
        prompt_version=prompt_version,
        clinician_summary=clinician_summary,
        patient_story=patient_story,
        disclaimer=disclaimer,
        generated_at=generated_at,
    )


def get_story_by_fingerprint(patient_id: str, fingerprint: str) -> Optional[PatientStoryResponse]:
    # Memory first
    cached = _cache_by_fingerprint.get((patient_id, fingerprint))
    if cached:
        return cached

    conn = _get_conn()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT patient_id, fingerprint, model_name, prompt_version,
                   clinician_summary, patient_story, disclaimer, generated_at
            FROM story_cache
            WHERE patient_id=%s AND fingerprint=%s
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (patient_id, fingerprint),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            story = _row_to_story(row)
            _cache_by_fingerprint[(patient_id, fingerprint)] = story
            _latest_by_patient[patient_id] = story
            return story
    except Exception as exc:  # noqa: BLE001
        print(f"[DB] get_story_by_fingerprint failed: {exc}")
    return None


def get_latest_story(patient_id: str) -> Optional[PatientStoryResponse]:
    cached = _latest_by_patient.get(patient_id)
    if cached:
        return cached

    conn = _get_conn()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT patient_id, fingerprint, model_name, prompt_version,
                   clinician_summary, patient_story, disclaimer, generated_at
            FROM story_cache
            WHERE patient_id=%s
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (patient_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            story = _row_to_story(row)
            _cache_by_fingerprint[(patient_id, story.fingerprint)] = story
            _latest_by_patient[patient_id] = story
            return story
    except Exception as exc:  # noqa: BLE001
        print(f"[DB] get_latest_story failed: {exc}")
    return None


def save_story(story: PatientStoryResponse) -> None:
    _cache_by_fingerprint[(story.patient_id, story.fingerprint)] = story
    _latest_by_patient[story.patient_id] = story

    conn = _get_conn()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO story_cache (
                patient_id, fingerprint, model_name, prompt_version,
                clinician_summary, patient_story, disclaimer, generated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                model_name=VALUES(model_name),
                prompt_version=VALUES(prompt_version),
                clinician_summary=VALUES(clinician_summary),
                patient_story=VALUES(patient_story),
                disclaimer=VALUES(disclaimer),
                generated_at=VALUES(generated_at)
            """,
            (
                story.patient_id,
                story.fingerprint,
                story.model_name,
                story.prompt_version,
                story.clinician_summary,
                story.patient_story,
                story.disclaimer,
                story.generated_at,
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[DB] save_story failed, kept in memory only: {exc}")
