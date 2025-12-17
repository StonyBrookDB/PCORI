# Week 2 – Patient Story API Specification

## 1) Overview
Service that takes **de-identified patient history** + **opioid risk info** and returns:
- Clinician-oriented summary
- Patient-friendly story

Models: primary GPT-4.1, fallback Claude Sonnet 4.5 (stubbed in Week 2).  
Base path: `/api/v1`  
Disclaimer: “This AI-generated summary is for educational support only and is not a substitute for professional medical judgment.”

## 2) Endpoints
### POST `/api/v1/story`
Generate or fetch cached story.
- Request: `PatientStoryRequest` (JSON)
- Response: `PatientStoryResponse` (JSON)
- Logic: canonicalize request → SHA256 fingerprint → lookup `(patient_id, fingerprint)` → return cached if hit; else call LLM, store, return.
- Errors: `400` invalid schema, `500` internal/LLM failure

### GET `/api/v1/story/{patient_id}`
Fetch most recent stored story (no regeneration).
- Response: last `PatientStoryResponse` for `patient_id`, or `404` if none.

### GET `/healthz`
Simple health check.

## 3) Schemas
### `PatientStoryRequest`
```jsonc
{
  "patient_id": "P12345",
  "age": 56,
  "sex": "male",
  "encounters": [
    {
      "date": "2024-11-01",
      "type": "outpatient",
      "diagnoses": ["chronic low back pain", "hypertension"],
      "medications": [
        {"name": "oxycodone", "dose": "10mg", "frequency": "BID"},
        {"name": "lisinopril", "dose": "20mg", "frequency": "OD"}
      ],
      "labs": [{"name": "eGFR", "value": 60, "unit": "mL/min/1.73m2"}],
      "notes": "Increased opioid use in past 3 months."
    }
  ],
  "opioid_risk": {
    "risk_score": 0.78,
    "risk_level": "high",
    "risk_factors": ["concurrent benzodiazepine use", "history of depression"]
  },
  "language": "en",
  "prompt_version": "v1"
}
```

### `PatientStoryResponse`
```jsonc
{
  "patient_id": "P12345",
  "fingerprint": "sha256-abcdef1234...",
  "model_name": "gpt-4.1",
  "prompt_version": "v1",
  "generated_at": "2025-11-23T10:15:00Z",
  "clinician_summary": "Concise clinical summary...",
  "patient_story": "Plain-language story...",
  "disclaimer": "This AI-generated summary is for educational support only and is not a substitute for professional medical judgment."
}
```

## 4) Fingerprint & Cache
- Fingerprint: SHA256 of canonical (sorted) JSON request, includes `prompt_version`.
- Cache key: `(patient_id, fingerprint)`; latest pointer per patient for quick fetch.
- Store fields: patient_id, fingerprint, model_name, prompt_version, clinician_summary, patient_story, disclaimer, generated_at, status (implicit in Week 2 stub).

## 5) Errors & Auth (Week 2 scope)
- `400` validation errors (handled by Pydantic/FastAPI)
- `404` when story not found (GET)
- `500` on internal/LLM errors (stubbed)
- Auth/rate limits: out of scope for Week 2; add headers/bearer later.
