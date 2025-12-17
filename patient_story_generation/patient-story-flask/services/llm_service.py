import json
from typing import Tuple

import requests

from config import Config

cfg = Config()


def build_prompt(payload: dict, audience: str) -> str:
    header = (
        "You are a clinical summarization assistant.\n"
        "Use ONLY the provided de-identified patient facts.\n"
        "Do NOT invent diagnoses, medications, or labs.\n"
    )
    if audience == "clinician":
        header += "Provide a concise clinical summary (200-300 words) with key risks.\n\n"
    else:
        header += "Provide a patient-friendly story (200-300 words), supportive tone.\n\n"
    facts = json.dumps(payload, indent=2)
    return header + "Patient facts:\n" + facts


def call_gemini(prompt: str) -> Tuple[str, dict]:
    if not cfg.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    import google.generativeai as genai

    genai.configure(api_key=cfg.gemini_api_key)
    model = genai.GenerativeModel(cfg.gemini_model)
    resp = model.generate_content(prompt)
    return resp.text or "", {"provider": "gemini", "model": cfg.gemini_model}


def call_openai(prompt: str) -> Tuple[str, dict]:
    if not cfg.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    # Azure OpenAI if api_base is set, else standard OpenAI
    is_azure = bool(cfg.openai_api_base)
    if is_azure:
        endpoint = cfg.openai_api_base.rstrip("/") + f"/openai/deployments/{cfg.openai_model}/chat/completions"
        params = {"api-version": cfg.openai_api_version or "2024-08-01-preview"}
        headers = {
            "Content-Type": "application/json",
            "api-key": cfg.openai_api_key,
        }
    else:
        endpoint = "https://api.openai.com/v1/chat/completions"
        params = {}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.openai_api_key}",
        }

    if is_azure:
        data = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1200}
    else:
        data = {"model": cfg.openai_model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1200}
    resp = requests.post(endpoint, params=params, headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    txt = j["choices"][0]["message"]["content"]
    usage = j.get("usage", {})
    return txt, {"provider": "openai", "model": cfg.openai_model, "usage": usage}


def call_anthropic(prompt: str) -> Tuple[str, dict]:
    if not cfg.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": cfg.anthropic_api_key,
        "anthropic-version": "2023-06-01",
    }
    data = {"model": cfg.anthropic_model, "max_tokens": 2048, "messages": [{"role": "user", "content": prompt}]}
    resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    txt = "".join([c.get("text", "") for c in j.get("content", [])])
    usage = j.get("usage", {})
    return txt, {"provider": "anthropic", "model": cfg.anthropic_model, "usage": usage}


def generate_story(payload: dict) -> Tuple[str, str, dict]:
    clinician_prompt = build_prompt(payload, "clinician")
    patient_prompt = build_prompt(payload, "patient")
    if cfg.primary_llm_provider == "gemini":
        clinician, meta_c = call_gemini(clinician_prompt)
        patient, meta_p = call_gemini(patient_prompt)
    elif cfg.primary_llm_provider == "openai":
        clinician, meta_c = call_openai(clinician_prompt)
        patient, meta_p = call_openai(patient_prompt)
    elif cfg.primary_llm_provider == "anthropic":
        clinician, meta_c = call_anthropic(clinician_prompt)
        patient, meta_p = call_anthropic(patient_prompt)
    else:
        raise RuntimeError(f"Unsupported provider {cfg.primary_llm_provider}")

    # merge metadata
    token_usage = {}
    if meta_c.get("usage"):
        token_usage["clinician"] = meta_c["usage"]
    if meta_p.get("usage"):
        token_usage["patient"] = meta_p["usage"]
    meta = {
        "provider": meta_c.get("provider"),
        "model": meta_c.get("model"),
        "token_usage": token_usage,
    }
    return clinician, patient, meta
