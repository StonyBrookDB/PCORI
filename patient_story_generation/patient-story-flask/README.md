# Patient Story Generation API (Flask)

End-to-end Flask service that generates:

- **Clinician Summary** (technical, concise)
- **Patient Story** (plain language, supportive)

from **de-identified** patient facts. It supports:

- **Patient ID only** input (fetches facts from `pcori_dashboard` MySQL tables)
- **Full JSON** input (skips DB fetch; still validates + caches)
- **Fingerprint-based caching** (avoid re-calling the LLM when nothing changed)
- **History mode** (multiple versions per patient when the fingerprint changes)
- **Basic PHI guardrails** (input + output scanning)

---

## Architecture (End-to-End Flow)

1. **Client call** (Postman / UI)
   - Either:
     - `POST /v1/patients/<patient_id>/generate` (patient_id only), or
     - `POST /v1/generate_story` (full JSON body)
2. **Input validation**
   - Validate required keys exist and types are reasonable.
3. **PHI checks (guardrails)**
   - Reject if common PHI keys or patterns are detected (e.g., email/phone/SSN-like patterns).
4. **Normalize payload**
   - Canonicalize/normalize the JSON so fingerprinting is stable and consistent.
5. **Fingerprint**
   - SHA256 hash of sorted canonical JSON.
6. **Cache lookup (history-aware)**
   - Lookup **by `(patient_id, fingerprint)`**.
   - Cache hit → return stored story (no LLM call).
7. **LLM call (on cache miss)**
   - Build prompts (clinician + patient).
   - Call configured provider (Gemini / OpenAI / **Azure OpenAI**).
8. **Output checks**
   - Sanitize output to reduce accidental PHI leakage.
9. **Persist story**
   - Insert a new row into `story_cache` (history mode; keeps old rows).
10. **Return response**
   - Includes `from_cache`, `fingerprint`, `model_name`, and optional `token_usage`.

---

## Repository Layout

```
patient-story-flask/
  main.py                      # Flask app + endpoints
  config.py                    # Environment-based configuration
  requirements.txt             # Python deps
  patient_story_request.schema.json
  services/
    data_service.py            # DB → build canonical patient JSON
    cache_service.py           # in-memory + MySQL story_cache logic
    llm_service.py             # prompts + LLM provider clients
  utils/
    validators.py              # schema validation, PHI checks, fingerprint, normalization
    response_formatter.py      # API response formatting
  sample_data/
    sample_patients.json       # sample JSON input (UI mode)
  test_api_key.py              # quick API key test (OpenAI or Azure OpenAI)
  test_gemini.py               # quick Gemini key test
  test_openai.py               # quick OpenAI key test
  test_azure_openai.py         # optional OpenAI SDK-based Azure test
```

---

## Requirements

- **Python**: 3.10+ recommended (works with newer versions too)
- **Pip** available
- Optional for DB mode:
  - Access to `pcori_dashboard` MySQL database
  - If remote DB requires SSH tunneling: an SSH client (`ssh`) and an open local port for forwarding

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration (`.env`)

Create `patient-story-flask/.env` (copy from `.env.example`) and fill in real values.

Key flags:

- `PRIMARY_LLM_PROVIDER`: `gemini` | `openai` | `anthropic`
- `USE_DB_CACHE`: `true` or `false`
- `LOG_LEVEL`: `INFO` (default) or `DEBUG` for verbose logs

### Option A — Azure OpenAI (recommended for production/demo)

Set `PRIMARY_LLM_PROVIDER=openai`, and set Azure/OpenAI fields:

```env
PRIMARY_LLM_PROVIDER=openai

# Azure OpenAI uses OPENAI_* variables in this codebase
OPENAI_API_KEY=__YOUR_AZURE_OPENAI_KEY__
OPENAI_API_BASE=https://YOUR-RESOURCE-NAME.cognitiveservices.azure.com/
OPENAI_API_VERSION=2025-01-01-preview

# IMPORTANT: for Azure, OPENAI_MODEL must be your DEPLOYMENT name
OPENAI_MODEL=gpt-5-chat
```

### Option B — Gemini

```env
PRIMARY_LLM_PROVIDER=gemini
GEMINI_API_KEY=__YOUR_GEMINI_KEY__
GEMINI_MODEL=gemini-2.0-flash
```

### Option C — OpenAI (standard)

```env
PRIMARY_LLM_PROVIDER=openai
OPENAI_API_KEY=__YOUR_OPENAI_KEY__
OPENAI_API_BASE=
OPENAI_API_VERSION=
OPENAI_MODEL=gpt-4.1
```

---

## Database Setup (pcori_dashboard)

### 1) DB connection settings (local DB or SSH tunnel)

These environment variables are used by the service:

```env
USE_DB_CACHE=true
DB_HOST=127.0.0.1
DB_PORT=23306
DB_USER=reddy
DB_PASSWORD=__YOUR_DB_PASSWORD__
DB_NAME=pcori_dashboard
STORY_CACHE_TABLE=story_cache
```

If your DB is remote, you typically create an SSH tunnel that forwards a **local port** (example: `23306`)
to the remote MySQL (`127.0.0.1:3306` on the remote server).

### Tables used to build the patient JSON

When you call `POST /v1/patients/{patient_id}/generate`, the service builds the de-identified JSON by querying:

- `hf_encounter` (drives the timeline; provides `ENCOUNTER_ID`, dates, encounter type, age, etc.)
- `hf_diagnosis` (diagnoses per encounter)
- `hf_medication` (medications per encounter)
- `hf_lab_procedure` (labs per encounter)
- `hf_clinical_event` (clinical events per encounter)
- `hf_procedure` (procedures per encounter)
- `hf_microbiology` (microbiology per encounter)
- `t_patient` (demographics such as race/gender/marital_status)
- `t_MME`, `t_prediction_od`, `t_prediction_oud`, `t_topfeature` (opioid risk signals)

### 2) SSH tunnel (Windows PowerShell example)

Run this in **a separate terminal** and keep it open:

```powershell
ssh -L 23306:127.0.0.1:3306 -p 130 <ssh_user>@130.245.130.189
```

Then your app connects to:

- `DB_HOST=127.0.0.1`
- `DB_PORT=23306`

### 3) story_cache table (history mode)

This service assumes a `story_cache` table exists in `pcori_dashboard`.

Suggested schema (history mode: multiple rows per patient over time):

```sql
CREATE TABLE IF NOT EXISTS pcori_dashboard.story_cache (
  id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  patient_id BIGINT NOT NULL,
  fingerprint VARCHAR(64) NOT NULL,
  model_name VARCHAR(128) NULL,
  prompt_version VARCHAR(32) NULL,
  clinician_summary LONGTEXT NULL,
  patient_story LONGTEXT NULL,
  disclaimer TEXT NULL,
  generated_at DATETIME NOT NULL,
  token_usage JSON NULL,
  UNIQUE KEY uniq_patient_fingerprint (patient_id, fingerprint),
  KEY idx_patient_generated_at (patient_id, generated_at),
  KEY idx_fingerprint (fingerprint)
);
```

---

## How to Run (Terminal)

### 1) Activate venv (Windows PowerShell)

```powershell
cd "C:\Users\mudiy\Documents\Advanced project\patient-story-flask"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
```

### 2) Start the Flask server

```powershell
python main.py
```

Server should start on:

- `http://127.0.0.1:5000`

Health check:

```powershell
curl.exe http://127.0.0.1:5000/healthz
```

---

## Endpoints (API Reference)

### 1) `GET /healthz`

Purpose: quick health check.

Response:

```json
{ "status": "ok" }
```

---

### 2) `POST /v1/patients/{patient_id}/generate`  (patient_id only → DB fetch)

Purpose: fetch patient facts from MySQL tables → fingerprint → cache → LLM → store story.

Request:

- URL: `http://127.0.0.1:5000/v1/patients/122665/generate`
- Method: `POST`
- Body: **none**
- Headers: none required (optional `Content-Type`)

Response (example keys):

```json
{
  "patient_id": "122665",
  "fingerprint": "…",
  "model_name": "gpt-5-chat",
  "generated_at": "2025-12-16 23:01:01",
  "from_cache": false,
  "clinician_summary": "…",
  "patient_story": "…",
  "disclaimer": "…",
  "token_usage": {}
}
```

Notes:

- If patient_id is not found in DB → `404 { "error": "patient_id not found" }`
- If `(patient_id, fingerprint)` already exists in `story_cache` → returns cached story with `from_cache=true`
- If fingerprint changed (data changed) → calls LLM and inserts a **new history row**

---

### 3) `POST /v1/generate_story` (full JSON → no DB fetch)

Purpose: UI provides full (de-identified) JSON payload → fingerprint → cache → LLM → store story.

If `USE_DB_CACHE=true`, this endpoint also verifies the `patient_id` exists in DB before generating.

Request:

- URL: `http://127.0.0.1:5000/v1/generate_story`
- Method: `POST`
- Headers:
  - `Content-Type: application/json`
- Body: JSON (see “Input Payload” below)

---

### 4) `GET /v1/patients/{patient_id}/history?limit=50`

Purpose: return all story versions for a patient (newest first).

Request:

- URL: `http://127.0.0.1:5000/v1/patients/122665/history?limit=50`
- Method: `GET`

Response:

```json
{
  "patient_id": "122665",
  "count": 2,
  "stories": [
    { "id": 10, "patient_id": "122665", "fingerprint": "…", "from_cache": true, ... },
    { "id": 7,  "patient_id": "122665", "fingerprint": "…", "from_cache": true, ... }
  ]
}
```

---

## Input Payload (Full JSON Mode)

For `POST /v1/generate_story`, use the same structure as the DB-generated JSON.

Minimal required keys:

- `patient_id`
- `age`
- `sex`
- `encounters` (array)
- `opioid_risk` (object)

Reference schema:

- `patient_story_request.schema.json`

Example payload:

```json
{
  "patient_id": "122665",
  "age": 26,
  "sex": "Male",
  "race": "Asian",
  "marital_status": "single",
  "language": "en",
  "prompt_version": "v1",
  "opioid_risk": {
    "risk_score": 0,
    "risk_level": "unknown",
    "risk_factors": [],
    "prediction_od": null,
    "mme_scores": []
  },
  "encounters": [
    {
      "date": "2011-08-04T13:57:00",
      "type": "Outpatient",
      "notes": "",
      "diagnoses": [
        { "code": "305.50", "description": "OPIOID ABUSE, UNSPECIFIED USE", "priority": 2, "type": "ICD9" }
      ],
      "medications": [],
      "labs": [],
      "events": [],
      "procedures": [],
      "microbiology": []
    }
  ]
}
```

---

## Running Without DB (Optional)

If you want to test the API without any MySQL access:

1. Set `USE_DB_CACHE=false` in `patient-story-flask/.env`
2. Start the server: `python main.py`
3. Use `POST /v1/generate_story` (full JSON mode)

Notes:

- `POST /v1/patients/{patient_id}/generate` requires DB access (it fetches patient facts).
- Cache will be **in-memory only** (cleared when the server restarts).
- When DB cache is disabled, `POST /v1/generate_story` will not verify patient existence in the database.

---

## Postman Instructions (Step-by-Step)

### A) Generate from patient_id only (DB fetch)

1. Method: `POST`
2. URL: `http://127.0.0.1:5000/v1/patients/122665/generate`
3. Headers: (optional) `Content-Type: application/json`
4. Body: **None**
5. Click **Send**

Expected:

- First call: `from_cache=false` and a new row inserted into `story_cache`
- Next call with no DB data changes: `from_cache=true` (cache hit)

### B) Generate from full JSON (no DB fetch)

1. Method: `POST`
2. URL: `http://127.0.0.1:5000/v1/generate_story`
3. Headers:
   - `Content-Type: application/json`
4. Body:
   - Choose **raw** → **JSON**
   - Paste the payload (see “Input Payload” above)
5. Click **Send**

If `USE_DB_CACHE=true` and the `patient_id` does not exist in DB, you will get:

```json
{ "error": "patient_id not found in database" }
```

### C) Fetch all story versions for a patient

1. Method: `GET`
2. URL: `http://127.0.0.1:5000/v1/patients/122665/history?limit=50`
3. Click **Send**

---

## PowerShell / curl Notes (Windows)

PowerShell aliases `curl` to `Invoke-WebRequest`, which does **not** accept `-X` like Linux curl.

Use:

- `curl.exe` (real curl), or
- `Invoke-RestMethod` / `Invoke-WebRequest`

Examples:

```powershell
# patient_id-only endpoint
curl.exe --% -X POST "http://127.0.0.1:5000/v1/patients/122665/generate"

# full JSON endpoint from a file
curl.exe --% -X POST "http://127.0.0.1:5000/v1/generate_story" -H "Content-Type: application/json" --data-binary "@payload.json"
```

---

## Logging / Debugging

### Log level

Set in `.env`:

```env
LOG_LEVEL=INFO
```

Use `DEBUG` to see verbose logs.

### Print the DB-generated payload (before LLM call)

Set:

```env
DEBUG_LOG_PAYLOAD=true
LOG_LEVEL=DEBUG
DEBUG_LOG_PAYLOAD_MAX_CHARS=20000
```

Then run `python main.py` and trigger `POST /v1/patients/<id>/generate`.
The payload will appear in the server terminal as a debug log entry.

---

## Caching Rules (History Mode)

Cache key is:

- `(patient_id, fingerprint)`

Behavior:

- **Same fingerprint** for a patient → cache hit, no LLM call, no new DB insert.
- **New fingerprint** for same patient → cache miss → LLM call → insert a new history row.

---

## PHI / Safety Notes

This system is designed for **de-identified** data only.

- Input PHI detection rejects common PHI patterns (email/phone/SSN-like strings) and suspicious keys.
- Output is sanitized as a last-resort guardrail.

You should still treat the LLM output as **supportive drafting**:

- Outputs must be reviewed by a clinician / instructor before clinical use.
