import json
import logging
import time
from datetime import datetime
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import Config
from services.cache_service import get_by_fingerprint, get_history, save
from services.data_service import fetch_patient_payload, patient_exists
from services.llm_service import generate_story
from utils.validators import validate_schema, check_phi, fingerprint, normalize_payload, sanitize_output_text
from utils.response_formatter import format_response

cfg = Config()

logging.basicConfig(
    level=getattr(logging, cfg.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def process_payload(payload: Dict[str, Any]) -> Dict:
    validate_schema(payload)
    check_phi(payload)
    payload = normalize_payload(payload)
    fp = fingerprint(payload)

    if cfg.debug_log_payload:
        try:
            payload_str = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            payload_str = str(payload)
        max_chars = max(0, int(cfg.debug_log_payload_max_chars))
        if max_chars and len(payload_str) > max_chars:
            payload_str = payload_str[:max_chars] + "\n... (truncated)"
        logger.debug("[PAYLOAD] patient_id=%s fingerprint=%s\n%s", payload.get("patient_id", ""), fp, payload_str)

    cached = get_by_fingerprint(payload.get("patient_id", ""), fp)
    if cached:
        return format_response(cached, True)

    start = time.perf_counter()
    clinician, patient, meta = generate_story(payload)
    latency_ms = (time.perf_counter() - start) * 1000

    clinician = sanitize_output_text(clinician)
    patient = sanitize_output_text(patient)

    record = {
        "patient_id": payload.get("patient_id", ""),
        "fingerprint": fp,
        "model_name": meta.get("model"),
        "prompt_version": payload.get("prompt_version", "v1"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "clinician_summary": clinician,
        "patient_story": patient,
        "disclaimer": cfg.disclaimer,
        "token_usage": meta.get("token_usage", {}),
        "latency_ms": latency_ms,
    }
    save(record)
    return format_response(record, False)


@app.route("/v1/generate_story", methods=["POST"])
def generate_story_full():
    try:
        payload = request.get_json(force=True)
        pid = str(payload.get("patient_id", ""))
        if cfg.use_db_cache and (not pid or not patient_exists(pid)):
            return jsonify({"error": "patient_id not found in database"}), 404
        resp = process_payload(payload)
        return jsonify(resp), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled error in /v1/generate_story")
        return jsonify({"error": "internal error"}), 500


@app.route("/v1/patients/<patient_id>/generate", methods=["POST"])
def generate_story_from_db(patient_id: str):
    data = fetch_patient_payload(patient_id)
    if not data:
        return jsonify({"error": "patient_id not found"}), 404
    try:
        resp = process_payload(data)
        return jsonify(resp), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled error in /v1/patients/%s/generate", patient_id)
        return jsonify({"error": "internal error"}), 500


@app.route("/v1/patients/<patient_id>/history", methods=["GET"])
def patient_story_history(patient_id: str):
    try:
        limit_raw = request.args.get("limit", "50")
        limit = int(limit_raw)
    except Exception:  # noqa: BLE001
        limit = 50

    rows = get_history(patient_id, limit=limit)
    stories = []
    for r in rows:
        item = format_response(r, from_cache=True)
        item["id"] = r.get("id")
        stories.append(item)

    return jsonify({"patient_id": patient_id, "count": len(stories), "stories": stories}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=cfg.port, debug=(cfg.flask_env == "development"))
