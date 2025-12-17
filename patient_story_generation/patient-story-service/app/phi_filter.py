"""Lightweight PHI filters to block obvious identifiers before LLM calls."""

from __future__ import annotations

import re
from typing import Iterable

from fastapi import HTTPException, status

from .models import PatientStoryRequest

# Schema-level hard blocks (even if extra fields were allowed).
PROHIBITED_KEYS = {
    "name",
    "first_name",
    "last_name",
    "full_name",
    "address",
    "zip",
    "zipcode",
    "ssn",
    "social_security_number",
    "mrn",
    "medical_record_number",
    "phone",
    "email",
    "dob",
    "date_of_birth",
}

# Regexes for common PHI patterns in free text.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
MRN_RE = re.compile(r"\bMRN\s*[:#]?\s*\d{6,}\b", re.IGNORECASE)


def _check_prohibited_keys(data: dict) -> Iterable[str]:
    for key in data.keys():
        if key in PROHIBITED_KEYS:
            yield key


def _scan_text(text: str) -> Iterable[str]:
    if EMAIL_RE.search(text):
        yield "email"
    if PHONE_RE.search(text):
        yield "phone"
    if SSN_RE.search(text):
        yield "ssn"
    if MRN_RE.search(text):
        yield "mrn"


def enforce_no_phi(req: PatientStoryRequest, strict: bool = True) -> None:
    """Raise HTTP 400 if obvious PHI is present."""
    # Check prohibited top-level keys by inspecting the raw dict (model forbids extra, but be defensive).
    raw = req.model_dump()
    bad_keys = list(_check_prohibited_keys(raw))
    if bad_keys:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prohibited PHI fields present: {', '.join(sorted(set(bad_keys)))}",
        )

    # Scan free-text fields likely to contain PHI (notes).
    for encounter in req.encounters:
        if encounter.notes:
            hits = list(_scan_text(encounter.notes))
            if hits and strict:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Potential PHI detected in notes ({', '.join(sorted(set(hits)))}); remove or redact and retry.",
                )
            # If not strict, we could redact; for now Week 3 uses strict.
