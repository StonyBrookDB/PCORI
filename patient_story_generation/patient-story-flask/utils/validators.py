import json
import re
from typing import Any, Dict, List

REQUIRED_FIELDS = ["patient_id", "age", "sex", "encounters", "opioid_risk"]

PHI_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}

PROHIBITED_KEYS = {
    "name",
    "full_name",
    "first_name",
    "last_name",
    "address",
    "dob",
    "mrn",
    "ssn",
    "email",
    "phone",
}


def validate_schema(payload: Dict[str, Any]) -> None:
    for field in REQUIRED_FIELDS:
        if field not in payload:
            raise ValueError(f"Missing required field: {field}")


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize payload so fingerprints are stable across UI vs DB sources.
    - patient_id always string
    - age/risk_score/event/lab numeric values coerced to float when possible
    - common string fields coerced to string when present
    """

    def to_str(v: Any) -> Any:
        if v is None:
            return None
        return str(v)

    def to_float(v: Any) -> Any:
        if v is None:
            return None
        try:
            return float(v)
        except Exception:  # noqa: BLE001
            return v

    def to_int(v: Any) -> Any:
        if v is None:
            return None
        try:
            return int(v)
        except Exception:  # noqa: BLE001
            return v

    out: Dict[str, Any] = dict(payload)
    out["patient_id"] = to_str(out.get("patient_id", ""))
    out["age"] = to_float(out.get("age"))
    out["sex"] = to_str(out.get("sex"))

    for k in ("race", "marital_status", "language", "prompt_version"):
        if k in out:
            out[k] = to_str(out.get(k))

    opioid_risk = out.get("opioid_risk")
    if isinstance(opioid_risk, dict):
        r = dict(opioid_risk)
        r["risk_score"] = to_float(r.get("risk_score"))
        if "risk_level" in r:
            r["risk_level"] = to_str(r.get("risk_level"))
        if "prediction_od" in r:
            r["prediction_od"] = to_int(r.get("prediction_od"))
        if "risk_factors" in r and isinstance(r.get("risk_factors"), list):
            r["risk_factors"] = [to_str(x) for x in r.get("risk_factors") if x is not None]
        if "mme_scores" in r and isinstance(r.get("mme_scores"), list):
            scores = []
            for s in r.get("mme_scores") or []:
                if not isinstance(s, dict):
                    continue
                ss = dict(s)
                ss["mme"] = to_float(ss.get("mme"))
                if "encounter_id" in ss:
                    ss["encounter_id"] = to_int(ss.get("encounter_id"))
                if "encounter_date" in ss:
                    ss["encounter_date"] = to_str(ss.get("encounter_date"))
                scores.append(ss)
            r["mme_scores"] = scores
        out["opioid_risk"] = r

    encounters_out = []
    encounters = out.get("encounters") if isinstance(out.get("encounters"), list) else []
    for enc in encounters:
        if not isinstance(enc, dict):
            continue
        e = dict(enc)
        if "date" in e:
            e["date"] = to_str(e.get("date"))
        if "type" in e:
            e["type"] = to_str(e.get("type"))
        if "notes" in e:
            e["notes"] = to_str(e.get("notes"))

        if isinstance(e.get("diagnoses"), list):
            dx_out = []
            for d in e.get("diagnoses") or []:
                if not isinstance(d, dict):
                    continue
                dd = dict(d)
                if "code" in dd:
                    dd["code"] = to_str(dd.get("code"))
                if "description" in dd:
                    dd["description"] = to_str(dd.get("description"))
                if "priority" in dd:
                    dd["priority"] = to_int(dd.get("priority"))
                if "type" in dd:
                    dd["type"] = to_str(dd.get("type"))
                dx_out.append(dd)
            e["diagnoses"] = dx_out

        if isinstance(e.get("events"), list):
            ev_out = []
            for v in e.get("events") or []:
                if not isinstance(v, dict):
                    continue
                vv = dict(v)
                if "code" in vv:
                    vv["code"] = to_str(vv.get("code"))
                if "time" in vv:
                    vv["time"] = to_str(vv.get("time"))
                if "value" in vv:
                    vv["value"] = to_float(vv.get("value"))
                ev_out.append(vv)
            e["events"] = ev_out

        if isinstance(e.get("labs"), list):
            labs_out = []
            for l in e.get("labs") or []:
                if not isinstance(l, dict):
                    continue
                ll = dict(l)
                if "code" in ll:
                    # keep as int when possible (these are typically procedure IDs)
                    ll["code"] = to_int(ll.get("code"))
                if "value" in ll:
                    ll["value"] = to_float(ll.get("value"))
                if "low" in ll:
                    ll["low"] = to_float(ll.get("low"))
                if "high" in ll:
                    ll["high"] = to_float(ll.get("high"))
                if "indicator" in ll:
                    ll["indicator"] = to_str(ll.get("indicator"))
                if "time" in ll:
                    ll["time"] = to_str(ll.get("time"))
                labs_out.append(ll)
            e["labs"] = labs_out

        if isinstance(e.get("medications"), list):
            meds_out = []
            for m in e.get("medications") or []:
                if not isinstance(m, dict):
                    continue
                mm = dict(m)
                for mk in ("name", "dose", "frequency", "start", "stop", "ndc"):
                    if mk in mm:
                        mm[mk] = to_str(mm.get(mk))
                meds_out.append(mm)
            e["medications"] = meds_out

        if isinstance(e.get("procedures"), list):
            procs_out = []
            for p in e.get("procedures") or []:
                if not isinstance(p, dict):
                    continue
                pp = dict(p)
                for pk in ("code", "description", "type", "time"):
                    if pk in pp:
                        pp[pk] = to_str(pp.get(pk))
                procs_out.append(pp)
            e["procedures"] = procs_out

        if isinstance(e.get("microbiology"), list):
            micro_out = []
            for mc in e.get("microbiology") or []:
                if not isinstance(mc, dict):
                    continue
                mcc = dict(mc)
                for mk in ("site", "time"):
                    if mk in mcc:
                        mcc[mk] = to_str(mcc.get(mk))
                micro_out.append(mcc)
            e["microbiology"] = micro_out

        encounters_out.append(e)

    out["encounters"] = encounters_out
    return out


def check_phi(payload: Dict[str, Any]) -> None:
    for key in payload.keys():
        if key.lower() in PROHIBITED_KEYS:
            raise ValueError(f"Prohibited PHI key: {key}")

    notes: List[str] = []
    encounters = payload.get("encounters", [])
    for e in encounters:
        if isinstance(e, dict) and e.get("notes"):
            notes.append(str(e.get("notes")))
    combined = " ".join(notes)

    for label, pattern in PHI_PATTERNS.items():
        if re.search(pattern, combined):
            raise ValueError(f"PHI detected ({label})")


def fingerprint(payload: Dict[str, Any]) -> str:
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def sanitize_output_text(text: str) -> str:
    """
    Basic output guardrail:
    - Redact strong PHI patterns if the model accidentally emits them.
    """
    if not text:
        return text
    redacted = text
    redacted = re.sub(PHI_PATTERNS["email"], "[EMAIL REDACTED]", redacted)
    redacted = re.sub(PHI_PATTERNS["phone"], "[PHONE REDACTED]", redacted)
    redacted = re.sub(PHI_PATTERNS["ssn"], "[SSN REDACTED]", redacted)
    return redacted
