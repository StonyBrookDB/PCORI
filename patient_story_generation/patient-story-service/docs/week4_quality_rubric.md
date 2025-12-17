# Week 4 – Prompt Refinement, Guardrails, and Quality Rubric

## Story style and tone
- Clinician Summary: 120–180 words, concise clinical tone, structured sentences (bullet-friendly), no invented data.
- Patient Story: 120–180 words, plain language, supportive tone, no jargon, actionable clarity.
- Headings required: “Clinician Summary:” and “Patient Story:”.
- Both sections under 200 words.

## Guardrails
- Input PHI scan: rejects requests with emails/phones/SSN/MRN or prohibited keys.
- Output guardrails: length cap (1200 chars/section), banned terms (e.g., suicide, SSN/MRN), PHI regex scan on outputs; rejects on violation.
- Caching/fingerprints unchanged (Week 2/3) for stability.

## Logging and observability
- Logs model name and latency per call.
- Cache hits/misses logged for `/api/v1/story`.

## Quality rubric (draft)
1) Factual grounding (0–3): No invented diagnoses/meds; aligns with provided data.
2) Clinical accuracy (0–3): Correct relationships (risk level, meds, labs), no unsafe advice.
3) Clarity & readability (0–3): Plain language in patient story; clinician summary concise.
4) Tone & empathy (0–2): Supportive, non-alarming language in patient story.
5) Safety & guardrails (0–2): No PHI leakage; length within bounds; banned terms absent.
6) Structure (0–2): Headings present; two clear sections; within target length.

## Sample story (patient_01) – clinician review notes
- Clinician Summary: Highlights chronic low back pain, hypertension, daily oxycodone use, worsening pain, high opioid risk (score 0.82), eGFR 60. Meets length/tone; no invented data observed.
- Patient Story: Plain language explanation of pain, blood pressure meds, daily opioid use, high risk; supportive tone; no PHI. Within length bounds.
- Guardrails passed: no PHI, banned words, or length violations; latency logged with model name `gemini-2.0-flash`.
