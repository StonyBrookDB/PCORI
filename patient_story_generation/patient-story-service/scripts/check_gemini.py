"""
Quick check to verify GEMINI_API_KEY is loaded and the LLM client can return a story.

Usage (from project root):
  python -m scripts.check_gemini

Requires:
- GEMINI_API_KEY set in environment (or via `--env-file .env` when running).
- LLM_PROVIDER set to `auto` or `gemini` to exercise the real call.

If the key is missing or the call fails, the script will fall back to the mock and print that status.
"""

import os
import sys
import json

from app.models import PatientStoryRequest
from app.llm_client import generate_story


def load_sample_payload() -> PatientStoryRequest:
    sample_path = os.path.join(os.path.dirname(__file__), "..", "examples", "patient_01.json")
    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PatientStoryRequest(**data)


def main() -> None:
    req = load_sample_payload()
    has_key = bool(os.getenv("GEMINI_API_KEY"))
    provider = os.getenv("LLM_PROVIDER", "auto")
    print(f"GEMINI_API_KEY present: {has_key}")
    print(f"LLM_PROVIDER: {provider}")

    clinician, patient, model_name = generate_story(req)
    print("\nResult:")
    print(f"model_name: {model_name}")
    print(f"clinician_summary: {clinician}")
    print(f"patient_story: {patient}")


if __name__ == "__main__":
    # Ensure project root on path when running as a module
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    main()
