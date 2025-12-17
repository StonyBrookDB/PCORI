"""
Negative test cases to verify guardrails and validation.

Run (from project root):
  LLM_PROVIDER=mock python -m scripts.test_negative
"""

import json

from fastapi.testclient import TestClient

from app.api import app


def load_payload(path="examples/patient_01.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_phi_block(client):
    payload = load_payload()
    payload["encounters"][0]["notes"] = "Call me at 555-123-4567"
    resp = client.post("/v1/generate_story", json=payload)
    assert resp.status_code == 400, resp.text
    assert "PHI" in resp.text or "phone" in resp.text


def test_extra_field_rejected(client):
    payload = load_payload()
    payload["unknown_field"] = "should_fail"
    resp = client.post("/v1/generate_story", json=payload)
    assert resp.status_code == 422, resp.text


def test_risk_score_range(client):
    payload = load_payload()
    payload["opioid_risk"]["risk_score"] = 1.5
    resp = client.post("/v1/generate_story", json=payload)
    assert resp.status_code == 422, resp.text


def test_bad_phi_file(client):
    payload = load_payload("examples/patient_bad_phi.json")
    resp = client.post("/v1/generate_story", json=payload)
    assert resp.status_code == 400, resp.text


def test_bad_risk_file(client):
    payload = load_payload("examples/patient_bad_risk.json")
    resp = client.post("/v1/generate_story", json=payload)
    assert resp.status_code == 422, resp.text


def test_bad_mrn_file(client):
    payload = load_payload("examples/patient_bad_mrn.json")
    resp = client.post("/v1/generate_story", json=payload)
    assert resp.status_code == 400, resp.text


def test_bad_name_phone_file(client):
    payload = load_payload("examples/patient_bad_name_phone.json")
    resp = client.post("/v1/generate_story", json=payload)
    assert resp.status_code == 400, resp.text


def test_bad_keys_file(client):
    payload = load_payload("examples/patient_bad_keys.json")
    resp = client.post("/v1/generate_story", json=payload)
    # Extra keys (name, phone) should trigger validation (422) or PHI block (400)
    assert resp.status_code in (400, 422), resp.text


if __name__ == "__main__":
    client = TestClient(app)
    test_phi_block(client)
    test_extra_field_rejected(client)
    test_risk_score_range(client)
    test_bad_phi_file(client)
    test_bad_risk_file(client)
    test_bad_mrn_file(client)
    test_bad_name_phone_file(client)
    test_bad_keys_file(client)
    print("Negative tests passed.")
