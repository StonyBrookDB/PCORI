import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import mysql.connector

from config import Config

cfg = Config()
logger = logging.getLogger(__name__)


def _conn():
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


def _fetch_all(cur, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    cur.execute(query, params)
    return [dict(row) for row in cur.fetchall()]


def _iso(val: Any) -> Any:
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, date):
        return val.isoformat()
    return val


def _num(val: Any) -> Any:
    if isinstance(val, Decimal):
        return float(val)
    return val


def fetch_patient_payload(patient_id: str) -> Optional[dict]:
    """
    Build the LLM input JSON from relational tables for the given patient_id.
    """
    conn = _conn()
    if not conn:
        return None

    try:
        cur = conn.cursor(dictionary=True)

        # Patient-level demographics
        patient_demo = {}
        cur.execute("SELECT race, gender, marital_status FROM t_patient WHERE patient_id=%s LIMIT 1", (patient_id,))
        row = cur.fetchone()
        if row:
            patient_demo = dict(row)

        # Encounters
        encounters = _fetch_all(
            cur,
            """
            SELECT ENCOUNTER_ID, ADMITTED_DT_TM, DISCHARGED_DT_TM,
                   PATIENT_TYPE_DESC, CARESETTING_DESC, DISCHG_DISP_CODE_DESC,
                   AGE_IN_YEARS, RACE, GENDER, MARITAL_STATUS
            FROM hf_encounter
            WHERE PATIENT_ID=%s
            ORDER BY ADMITTED_DT_TM
            """,
            (patient_id,),
        )
        if not encounters:
            cur.close()
            conn.close()
            return None

        enc_ids = [e["ENCOUNTER_ID"] for e in encounters]
        enc_param = tuple(enc_ids)

        # Diagnoses
        diagnoses: Dict[int, List[Dict[str, Any]]] = {}
        if enc_ids:
            for row in _fetch_all(
                cur,
                """
                SELECT ENCOUNTER_ID, DIAGNOSIS_CODE, DIAGNOSIS_DESCRIPTION, DIAGNOSIS_PRIORITY, DIAGNOSIS_TYPE
                FROM hf_diagnosis
                WHERE ENCOUNTER_ID IN (%s)
                """ % (",".join(["%s"] * len(enc_ids))),
                enc_param,
            ):
                diagnoses.setdefault(row["ENCOUNTER_ID"], []).append(
                    {
                        "code": row["DIAGNOSIS_CODE"],
                        "description": row.get("DIAGNOSIS_DESCRIPTION"),
                        "priority": row.get("DIAGNOSIS_PRIORITY"),
                        "type": row.get("DIAGNOSIS_TYPE"),
                    }
                )

        # Medications
        medications: Dict[int, List[Dict[str, Any]]] = {}
        if enc_ids:
            for row in _fetch_all(
                cur,
                """
                SELECT ENCOUNTER_ID, GENERIC_NAME, PRODUCT_STRENGTH_DESCRIPTION, ORDER_STRENGTH,
                       FREQUENCY_DESC, MED_STARTED_DT_TM, MED_STOPPED_DT_TM, NDC_CODE
                FROM hf_medication
                WHERE ENCOUNTER_ID IN (%s)
                """ % (",".join(["%s"] * len(enc_ids))),
                enc_param,
            ):
                strength = row.get("PRODUCT_STRENGTH_DESCRIPTION") or row.get("ORDER_STRENGTH")
                medications.setdefault(row["ENCOUNTER_ID"], []).append(
                    {
                        "name": row.get("GENERIC_NAME"),
                        "dose": str(_num(strength)) if strength is not None else None,
                        "frequency": row.get("FREQUENCY_DESC"),
                        "start": _iso(row.get("MED_STARTED_DT_TM")),
                        "stop": _iso(row.get("MED_STOPPED_DT_TM")),
                        "ndc": str(_num(row.get("NDC_CODE"))) if row.get("NDC_CODE") is not None else None,
                    }
                )

        # Labs
        labs: Dict[int, List[Dict[str, Any]]] = {}
        if enc_ids:
            for row in _fetch_all(
                cur,
                """
                SELECT ENCOUNTER_ID, DETAIL_LAB_PROCEDURE_ID, NUMERIC_RESULT,
                       NORMAL_RANGE_LOW, NORMAL_RANGE_HIGH, RESULT_INDICATOR_DESC, LAB_COMPLETED_DT_TM
                FROM hf_lab_procedure
                WHERE ENCOUNTER_ID IN (%s)
                """ % (",".join(["%s"] * len(enc_ids))),
                enc_param,
            ):
                labs.setdefault(row["ENCOUNTER_ID"], []).append(
                    {
                        "code": row.get("DETAIL_LAB_PROCEDURE_ID"),
                        "value": _num(row.get("NUMERIC_RESULT")),
                        "low": _num(row.get("NORMAL_RANGE_LOW")),
                        "high": _num(row.get("NORMAL_RANGE_HIGH")),
                        "indicator": row.get("RESULT_INDICATOR_DESC"),
                        "time": _iso(row.get("LAB_COMPLETED_DT_TM")),
                    }
                )

        # Clinical events
        events: Dict[int, List[Dict[str, Any]]] = {}
        if enc_ids:
            for row in _fetch_all(
                cur,
                """
                SELECT ENCOUNTER_ID, EVENT_CODE_DESC, EVENT_CODE_DISPLAY, RESULT_VALUE_NUM, VERIFIED_DT_TM
                FROM hf_clinical_event
                WHERE ENCOUNTER_ID IN (%s)
                """ % (",".join(["%s"] * len(enc_ids))),
                enc_param,
            ):
                events.setdefault(row["ENCOUNTER_ID"], []).append(
                    {
                        "code": row.get("EVENT_CODE_DESC") or row.get("EVENT_CODE_DISPLAY"),
                        "value": _num(row.get("RESULT_VALUE_NUM")),
                        "time": _iso(row.get("VERIFIED_DT_TM")),
                    }
                )

        # Procedures
        procedures: Dict[int, List[Dict[str, Any]]] = {}
        if enc_ids:
            for row in _fetch_all(
                cur,
                """
                SELECT ENCOUNTER_ID, PROCEDURE_CODE, PROCEDURE_DESCRIPTION, PROCEDURE_TYPE, PROCEDURE_DT_TM
                FROM hf_procedure
                WHERE ENCOUNTER_ID IN (%s)
                """ % (",".join(["%s"] * len(enc_ids))),
                enc_param,
            ):
                procedures.setdefault(row["ENCOUNTER_ID"], []).append(
                    {
                        "code": row.get("PROCEDURE_CODE"),
                        "description": row.get("PROCEDURE_DESCRIPTION"),
                        "type": row.get("PROCEDURE_TYPE"),
                        "time": _iso(row.get("PROCEDURE_DT_TM")),
                    }
                )

        # Microbiology
        micro: Dict[int, List[Dict[str, Any]]] = {}
        if enc_ids:
            for row in _fetch_all(
                cur,
                """
                SELECT ENCOUNTER_ID, COLLECTION_SOURCE_SITE_DESC, LAST_REPORT_UPDATED_DT_TM
                FROM hf_microbiology
                WHERE ENCOUNTER_ID IN (%s)
                """ % (",".join(["%s"] * len(enc_ids))),
                enc_param,
            ):
                micro.setdefault(row["ENCOUNTER_ID"], []).append(
                    {
                        "site": row.get("COLLECTION_SOURCE_SITE_DESC"),
                        "time": _iso(row.get("LAST_REPORT_UPDATED_DT_TM")),
                    }
                )

        # Risk signals
        cur.execute("SELECT label FROM t_prediction_oud WHERE patient_id=%s LIMIT 1", (patient_id,))
        oud = cur.fetchone()
        cur.execute("SELECT label FROM t_prediction_od WHERE patient_id=%s LIMIT 1", (patient_id,))
        od = cur.fetchone()
        cur.execute(
            "SELECT encounter_id, encounter_date, mme_score FROM t_MME WHERE patient_id=%s", (patient_id,)
        )
        mme_rows = cur.fetchall() or []
        cur.execute(
            """
            SELECT feature_name, ranking, score
            FROM t_topfeature
            WHERE patient_id=%s
            ORDER BY ranking ASC
            LIMIT 10
            """,
            (patient_id,),
        )
        top_features = cur.fetchall() or []

        cur.close()
        conn.close()

        def risk_level(label: Optional[int]) -> str:
            if label is None:
                return "unknown"
            return "high" if label == 1 else "low"

        opioid_risk = {
            "risk_score": (oud["label"] if oud else 0) or 0,
            "risk_level": risk_level(oud["label"] if oud else None),
            "risk_factors": [tf["feature_name"] for tf in top_features],
            "prediction_od": od["label"] if od else None,
            "mme_scores": [
                {
                    "encounter_id": r.get("encounter_id"),
                    "encounter_date": _iso(r.get("encounter_date")),
                    "mme": _num(r.get("mme_score")),
                }
                for r in mme_rows
            ],
        }

        # Build encounter list
        encounter_objs = []
        for e in encounters:
            eid = e["ENCOUNTER_ID"]
            encounter_objs.append(
                {
                    "date": _iso(e.get("ADMITTED_DT_TM")),
                    "type": e.get("PATIENT_TYPE_DESC") or e.get("CARESETTING_DESC"),
                    "diagnoses": diagnoses.get(eid, []),
                    "medications": medications.get(eid, []),
                    "labs": labs.get(eid, []),
                    "events": events.get(eid, []),
                    "procedures": procedures.get(eid, []),
                    "microbiology": micro.get(eid, []),
                    "notes": "",
                }
            )

        # Compose final payload for LLM
        payload = {
            "patient_id": patient_id,
            "age": _num(encounters[0].get("AGE_IN_YEARS")),
            "sex": patient_demo.get("gender") or encounters[0].get("GENDER"),
            "race": patient_demo.get("race") or encounters[0].get("RACE"),
            "marital_status": patient_demo.get("marital_status") or encounters[0].get("MARITAL_STATUS"),
            "language": "en",
            "prompt_version": "v1",
            "opioid_risk": opioid_risk,
            "encounters": encounter_objs,
        }
        return payload
    except Exception:  # noqa: BLE001
        logger.exception("fetch_patient_payload failed for patient_id=%s", patient_id)
        return None


def patient_exists(patient_id: str) -> bool:
    """
    Return True if patient_id exists in the source DB (pcori_dashboard).
    We treat existence as: at least one encounter row is present.
    """
    conn = _conn()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM hf_encounter WHERE PATIENT_ID=%s LIMIT 1", (patient_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return bool(row)
    except Exception:  # noqa: BLE001
        logger.exception("patient_exists failed for patient_id=%s", patient_id)
        return False
