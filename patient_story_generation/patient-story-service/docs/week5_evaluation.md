# Week 5 – Evaluation, Guardrails Check, and Demo Notes

## Evaluation plan (target ≥30 cases)
- Sources: use the existing sample JSONs plus additional de-identified cases to reach ≥30.
- Metrics (score 0–3 unless noted):
  1) Readability (clinician, patient)
  2) Factual alignment (no invented data)
  3) Safety/guardrails (PHI, banned terms, length)
  4) Tone/empathy (patient section) (0–2)
  5) Structure/format (headings, length bounds) (0–2)
  6) Clinical accuracy (risk context, meds/labs) (0–3)
- Failure modes to note: hallucinated meds/diagnoses, missing key facts, tone too alarming, exceeds length, PHI leakage, banned terms.
- Record: keep a simple table (case id, pass/fail per rubric, notes).

## Optional endpoints (added)
- GET `/v1/patients/{id}/story` – fetch latest story (alias).
- POST `/v1/patients/{id}/regenerate` – force regeneration with provided request body (bypasses cache).

## Guardrails recap
- Input: PHI scan (emails/phones/SSN/MRN, prohibited keys).
- Output: length cap, banned terms, PHI regex scan; rejects on violation.
- Logging: model name and latency per call.

## Demo checklist (3–5 minutes)
- Start server with env file; show one generate call, one regenerate, one GET latest.
- Show guardrail: inject phone/email into notes → 400 input PHI block.
- (Optional) Show output guardrail by forcing banned term to prove rejection.
- Mention rubric and how ≥30 cases were scored; summarize failure modes observed.
