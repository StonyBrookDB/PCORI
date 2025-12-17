"""Helpers to compute stable JSON fingerprints for cache keys."""

import hashlib
import json

from .models import PatientStoryRequest


def compute_fingerprint(req: PatientStoryRequest) -> str:
    """
    Compute a SHA256 fingerprint of the request JSON with sorted keys.
    Any change in data or prompt_version produces a new fingerprint.
    """
    data = json.loads(req.json())
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
