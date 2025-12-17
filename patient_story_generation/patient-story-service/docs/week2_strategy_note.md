# Week 2 – Generation Strategy, Caching, and Regeneration Triggers

## Goal
Millions of patients → we cannot regenerate everything. Treat each story as a cached artifact tied to a fingerprint of the de-identified input and prompt version.

## Strategy: On-demand + Cache
1. Client requests a story for `patient_id`.
2. Backend builds `PatientStoryRequest` JSON from current data.
3. Canonicalize (sorted JSON) → SHA256 `fingerprint`.
4. Lookup `(patient_id, fingerprint)` in `patient_stories` (DB in prod, in-memory in Week 2).
   - Hit → return cached story.
   - Miss → call LLM, store with fingerprint, return.

Result: each unique state of the patient data is generated once; pay only when accessed.

## Cache Storage (concept)
- Table: `patient_stories`
- Columns: patient_id, fingerprint, model_name, prompt_version, clinician_summary, patient_story, disclaimer, generated_at, status ("ok"/"error")
- Latest pointer per patient for quick fetch.
- Week 2: simulated with dicts in `app/cache.py`.

## Regeneration Triggers
1) **Data/prompt change (automatic)**  
Any change to encounters, meds, labs, risk, or prompt_version alters the canonical JSON → new SHA256 → cache miss → regenerate → store new row.

2) **Manual refresh (future endpoint)**  
`POST /api/v1/story/{patient_id}/regenerate` to bypass cache and force a new story with current data; store new fingerprint and mark as latest.

3) **Optional policies (future)**  
- TTL (e.g., regenerate if older than N months).  
- Precompute small cohorts (e.g., top-risk or recently active patients) to warm cache.

## Monitoring (future)
- Track cache hit rate, generation latency, error rate.
- Log fingerprint and prompt_version for traceability; avoid logging PHI.

## Summary
Generate on demand, cache by `(patient_id, fingerprint)`, regenerate only when data/prompt changes or when manually requested. Scales to millions of patients while controlling cost.
