"""LLM client that calls Gemini (no mock fallback)."""

from __future__ import annotations

import os
from typing import Tuple

from .models import PatientStoryRequest


def _build_prompt(req: PatientStoryRequest) -> str:
    """Create an instruction prompt emphasizing tone, style, and length."""
    lines = [
        "You are a clinical summarization assistant.",
        "Use ONLY the provided de-identified patient facts.",
        "Do NOT invent diagnoses, medications, or labs.",
        "Return two sections using these headings exactly:",
        "Clinician Summary: (120-180 words, concise, clinical tone, bullet-friendly sentences).",
        "Patient Story: (120-180 words, plain language, supportive tone).",
        "Keep each section under 200 words.",
        "",
        "Patient facts:",
        f"- Age: {req.age}",
        f"- Sex: {req.sex}",
        f"- Opioid risk: level={req.opioid_risk.risk_level}, score={req.opioid_risk.risk_score}",
    ]
    for idx, enc in enumerate(req.encounters, start=1):
        lines.append(f"- Encounter {idx}: date={enc.date}, type={enc.type}")
        if enc.diagnoses:
            lines.append(f"  Diagnoses: {', '.join(enc.diagnoses)}")
        if enc.medications:
            meds = "; ".join(f"{m.name} {m.dose} {m.frequency}" for m in enc.medications)
            lines.append(f"  Meds: {meds}")
        if enc.labs:
            labs = "; ".join(f"{l.name} {l.value}{l.unit}" for l in enc.labs)
            lines.append(f"  Labs: {labs}")
        if enc.notes:
            lines.append(f"  Notes: {enc.notes}")
    return "\n".join(lines)

def _gemini_response(req: PatientStoryRequest) -> Tuple[str, str, str]:
    import google.generativeai as genai  # type: ignore

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    model = genai.GenerativeModel(model_name)

    prompt = _build_prompt(req)
    response = model.generate_content(prompt)
    text = response.text or ""

    # Simple split: expect two sections separated by headings; fallback to whole text.
    clinician, patient = "", ""
    if "Patient Story:" in text:
        parts = text.split("Patient Story:", 1)
        clinician = parts[0].replace("Clinician Summary:", "").strip()
        patient = parts[1].strip()
    else:
        clinician = text.strip()
        patient = "Plain-language explanation not separated; using full text."

    return clinician, patient, model_name


def generate_story(req: PatientStoryRequest) -> Tuple[str, str, str]:
    """Generate clinician and patient stories via Gemini."""
    return _gemini_response(req)
