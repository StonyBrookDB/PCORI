# Patient Story Service — Week 2/3 Implementation

Week 2: schemas, disclaimer, on-demand + cache, strategy docs, samples, FastAPI skeleton.  
Week 3: `/v1/generate_story` with stricter validation, PHI checks.  
Week 4: refined prompt (tone/length), output guardrails (length, banned terms, PHI scan), logging of model/latency, quality rubric doc.  
Week 5: evaluation plan, optional patient endpoints (get/regenerate), demo checklist.  
Current: Gemini-only LLM client (no mock fallback) with optional MySQL-backed cache.

## Structure
- `app/` – FastAPI app, Pydantic models, fingerprint helper, cache (in-memory + optional MySQL), PHI filter, LLM client (Gemini only).
- `docs/` – Week 2 API spec (`week2_api_spec.md`), strategy note (`week2_strategy_note.md`), Week 4 quality rubric (`week4_quality_rubric.md`), Week 5 evaluation/demo notes (`week5_evaluation.md`).
- `examples/` – Six de-identified sample input JSONs.
- `requirements.txt` – fastapi, uvicorn[standard], pydantic, google-generativeai, httpx.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.api:app --reload
```

## Test endpoints
Week 2 cache endpoint:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/story \
  -H "Content-Type: application/json" \
  --data-binary @examples/patient_01.json
```

Repeat to see cache hit:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/story \
  -H "Content-Type: application/json" \
  --data-binary @examples/patient_01.json
```

Fetch latest for a patient:
```bash
curl http://127.0.0.1:8000/api/v1/story/P0001
```

Week 3 generate (no cache dependency):
```bash
curl -X POST http://127.0.0.1:8000/v1/generate_story \
  -H "Content-Type: application/json" \
  --data-binary @examples/patient_01.json
```

Week 5 DB fetch (if patient_data table populated):
```bash
curl -X POST http://127.0.0.1:8000/v1/patients/P0001/generate
```

Health check:
```bash
curl http://127.0.0.1:8000/healthz
```

## Week 2 scope
- Input/output JSON schemas with validation and disclaimer.
- Strategy: on-demand generation + cache keyed by `(patient_id, fingerprint)` (fingerprint = SHA256 of canonical request JSON, includes `prompt_version`).
- Regeneration triggers: data/prompt change → new fingerprint; manual endpoint/TTL optional future work.
- Stub LLM client (simulated output).

## Week 3 scope
- Endpoint: `POST /v1/generate_story` (uses same request/response schema).
- Validation tightened: age > 0, age <= 120, risk_score 0–1, enums for risk_level and encounter type, extra fields forbidden.
- PHI filter: blocks obvious identifiers (email/phone/SSN/MRN) and prohibited keys; rejects on detection (strict mode).
- LLM client: default mock; if `GEMINI_API_KEY` is set (and `LLM_PROVIDER` is `auto` or `gemini`), uses Gemini (`GEMINI_MODEL` defaults to `gemini-1.5-flash`). Falls back to mock on failure.

## Env vars (Gemini only, optional MySQL cache)
- `GEMINI_API_KEY` — set to your key to enable real calls.
- `GEMINI_MODEL` — optional, default `gemini-2.0-flash`.
- `USE_DB_CACHE` — `true` to enable MySQL persistence of stories.
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` — MySQL connection.

## Week 4 additions
- Prompt refined for tone/style/length (120–180 words per section; headings required).
- Output guardrails: length cap, banned terms, PHI scan on outputs.
- Logging: model name and latency per call.
- Quality rubric draft: see `docs/week4_quality_rubric.md`.

## Week 5 additions
- Optional endpoints: `GET /v1/patients/{id}/story` (alias) and `POST /v1/patients/{id}/regenerate` (bypass cache).
- Evaluation plan/rubric notes: see `docs/week5_evaluation.md` (target ≥30 cases; readability, factual alignment, safety, tone, structure).
- Demo checklist: show generate, regenerate, GET latest, and guardrail behavior.
- DB-backed fetch/generate: `POST /v1/patients/{id}/generate` if you store patient JSON in `patient_data.payload`.

Example `.env` (do not commit):
```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
USE_DB_CACHE=true
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=root
DB_NAME=patient_story
```

## MySQL table (for USE_DB_CACHE=true)
```sql
CREATE TABLE IF NOT EXISTS story_cache (
  id INT AUTO_INCREMENT PRIMARY KEY,
  patient_id VARCHAR(50) NOT NULL,
  fingerprint VARCHAR(64) NOT NULL UNIQUE,
  model_name VARCHAR(100) NOT NULL,
  prompt_version VARCHAR(50) NOT NULL,
  clinician_summary TEXT,
  patient_story TEXT,
  disclaimer TEXT,
  generated_at VARCHAR(50),
  INDEX idx_patient (patient_id),
  INDEX idx_generated (generated_at)
);

CREATE TABLE IF NOT EXISTS patient_data (
  patient_id VARCHAR(50) PRIMARY KEY,
  payload JSON NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```
