"""FastAPI entrypoint for the patient story service."""

from datetime import datetime, timezone
from time import perf_counter

from fastapi import FastAPI, HTTPException

from .cache import get_latest_story, get_story_by_fingerprint, save_story
from .data_service import fetch_patient_data
from .fingerprint import compute_fingerprint
from .llm_client import generate_story
from .models import PatientStoryRequest, PatientStoryResponse
from .phi_filter import enforce_no_phi
from .guardrails import enforce_output_guardrails

DISCLAIMER_TEXT = (
    "This AI-generated summary is for educational support only and is not a "
    "substitute for professional medical judgment."
)

app = FastAPI(
    title="Patient Story Service",
    version="0.1.0",
    description="Generates clinician summaries and patient stories from de-identified patient histories.",
)


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/story", response_model=PatientStoryResponse)
def create_or_get_story(req: PatientStoryRequest) -> PatientStoryResponse:
    """
    Generate or fetch cached story for a patient using (patient_id, fingerprint).
    """
    enforce_no_phi(req, strict=True)
    fingerprint = compute_fingerprint(req)

    cached = get_story_by_fingerprint(req.patient_id, fingerprint)
    if cached:
        print(f"[CACHE] Hit for patient {req.patient_id}")
        return cached

    start = perf_counter()
    clinician_summary, patient_story, model_name = generate_story(req)
    latency_ms = (perf_counter() - start) * 1000
    enforce_output_guardrails(clinician_summary, patient_story)

    new_story = PatientStoryResponse(
        patient_id=req.patient_id,
        fingerprint=fingerprint,
        model_name=model_name,
        prompt_version=req.prompt_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
        clinician_summary=clinician_summary,
        patient_story=patient_story,
        disclaimer=DISCLAIMER_TEXT,
    )

    save_story(new_story)
    print(f"[CACHE] Saved story for patient {req.patient_id}")
    print(f"[LLM] model={model_name} latency_ms={latency_ms:.1f} patient_id={req.patient_id}")

    return new_story


@app.get("/api/v1/story/{patient_id}", response_model=PatientStoryResponse)
def get_story(patient_id: str) -> PatientStoryResponse:
    """
    Fetch the most recent stored story for a patient.
    """
    cached = get_latest_story(patient_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Story not found for this patient_id")
    return cached


@app.get("/v1/patients/{patient_id}/story", response_model=PatientStoryResponse)
def get_story_v1(patient_id: str) -> PatientStoryResponse:
    """
    Alias for fetching the latest story for a patient (Week 5 optional endpoint).
    """
    cached = get_latest_story(patient_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Story not found for this patient_id")
    return cached


@app.post("/v1/generate_story", response_model=PatientStoryResponse)
def generate_story_v1(req: PatientStoryRequest) -> PatientStoryResponse:
    """
    Week 3 endpoint: validate, PHI-scan, call LLM (Gemini if configured, else mock).
    """
    enforce_no_phi(req, strict=True)
    fingerprint = compute_fingerprint(req)
    start = perf_counter()
    clinician_summary, patient_story, model_name = generate_story(req)
    latency_ms = (perf_counter() - start) * 1000
    enforce_output_guardrails(clinician_summary, patient_story)

    new_story = PatientStoryResponse(
        patient_id=req.patient_id,
        fingerprint=fingerprint,
        model_name=model_name,
        prompt_version=req.prompt_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
        clinician_summary=clinician_summary,
        patient_story=patient_story,
        disclaimer=DISCLAIMER_TEXT,
    )
    save_story(new_story)
    print(f"[LLM] model={model_name} latency_ms={latency_ms:.1f} patient_id={req.patient_id}")
    return new_story


@app.post("/v1/patients/{patient_id}/generate", response_model=PatientStoryResponse)
def generate_story_from_db(patient_id: str) -> PatientStoryResponse:
    """
    Fetch patient data from DB by patient_id, then generate story.
    """
    data = fetch_patient_data(patient_id)
    if not data:
        raise HTTPException(status_code=404, detail="No patient data found for this patient_id")
    try:
        req = PatientStoryRequest(**data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid patient data for {patient_id}: {exc}")

    return generate_story_v1(req)


@app.post("/v1/patients/{patient_id}/regenerate", response_model=PatientStoryResponse)
def regenerate_story(patient_id: str, req: PatientStoryRequest) -> PatientStoryResponse:
    """
    Force regeneration for a patient using the provided request body (bypasses cache).
    """
    if req.patient_id != patient_id:
        raise HTTPException(status_code=400, detail="patient_id in path and body must match")
    enforce_no_phi(req, strict=True)
    fingerprint = compute_fingerprint(req)
    start = perf_counter()
    clinician_summary, patient_story, model_name = generate_story(req)
    latency_ms = (perf_counter() - start) * 1000
    enforce_output_guardrails(clinician_summary, patient_story)

    new_story = PatientStoryResponse(
        patient_id=req.patient_id,
        fingerprint=fingerprint,
        model_name=model_name,
        prompt_version=req.prompt_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
        clinician_summary=clinician_summary,
        patient_story=patient_story,
        disclaimer=DISCLAIMER_TEXT,
    )
    save_story(new_story)
    print(f"[LLM] model={model_name} latency_ms={latency_ms:.1f} patient_id={req.patient_id} (regenerate)")
    return new_story
