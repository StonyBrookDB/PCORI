"""Output guardrails for generated stories (length, banned words, PHI scan)."""

from __future__ import annotations

import re
from typing import Iterable

from fastapi import HTTPException, status

from .phi_filter import EMAIL_RE, PHONE_RE, SSN_RE, MRN_RE


MAX_CHARS = 1200  # per section
BANNED_WORDS = {"suicide", "homicide", "kill yourself", "social security number", "ssn", "mrn"}


def _find_banned(text: str) -> Iterable[str]:
    lower = text.lower()
    for word in BANNED_WORDS:
        if word in lower:
            yield word


def _has_phi(text: str) -> Iterable[str]:
    hits = []
    if EMAIL_RE.search(text):
        hits.append("email")
    if PHONE_RE.search(text):
        hits.append("phone")
    if SSN_RE.search(text):
        hits.append("ssn")
    if MRN_RE.search(text):
        hits.append("mrn")
    return hits


def enforce_output_guardrails(clinician: str, patient: str) -> None:
    """Raise HTTP 500 if generated output violates guardrails."""
    for label, section in (("clinician_summary", clinician), ("patient_story", patient)):
        if len(section) > MAX_CHARS:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{label} exceeds length limit",
            )
        banned_hits = list(_find_banned(section))
        if banned_hits:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{label} contains banned terms: {', '.join(sorted(set(banned_hits)))}",
            )
        phi_hits = _has_phi(section)
        if phi_hits:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{label} appears to include PHI ({', '.join(sorted(set(phi_hits)))}); output rejected",
            )
