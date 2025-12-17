import time
import json
from typing import Dict

from config import Config

cfg = Config()


def format_response(record: Dict, from_cache: bool) -> Dict:
    token_usage = record.get("token_usage", {})
    if isinstance(token_usage, str):
        try:
            token_usage = json.loads(token_usage)
        except Exception:  # noqa: BLE001
            token_usage = {}

    out = {
        "patient_id": record.get("patient_id", ""),
        "fingerprint": record.get("fingerprint", ""),
        "model_name": record.get("model_name", ""),
        "prompt_version": record.get("prompt_version", "v1"),
        "generated_at": record.get("generated_at", time.strftime("%Y-%m-%dT%H:%M:%S%z")),
        "clinician_summary": record.get("clinician_summary", ""),
        "patient_story": record.get("patient_story", ""),
        "disclaimer": record.get("disclaimer", cfg.disclaimer),
        "from_cache": from_cache,
        "token_usage": token_usage,
    }
    return out
