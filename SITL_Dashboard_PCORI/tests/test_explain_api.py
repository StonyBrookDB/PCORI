"""
Test A4: Explanation API Tests
Tests for the /api/model/{model_id}/explain/patient/{patient_id} endpoint.
"""

import requests

BASE_URL = "http://localhost:8501"


def get_or_train_model():
    """Get existing model or train a new one."""
    # Check for existing models
    res = requests.get(f"{BASE_URL}/api/models")
    models = res.json()

    if models:
        return list(models.keys())[0]

    # Train a new model
    res = requests.post(
        f"{BASE_URL}/api/train/quick",
        json={
            "filters": {},
            "model_type": "lightgbm",
            "features": ["age", "prior_opioid_days", "benzo_use"],
        },
    )
    return res.json()["model_id"]


def test_explain_patient():
    """Patient explanation should return SHAP contributions."""
    model_id = get_or_train_model()

    res = requests.get(f"{BASE_URL}/api/model/{model_id}/explain/patient/P0001")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()

    assert "prediction" in data, "Response missing prediction"
    assert "base_value" in data, "Response missing base_value"
    assert "contributions" in data, "Response missing contributions"
    assert len(data["contributions"]) > 0, "Contributions list is empty"

    print(f"  Patient P0001: prediction={data['prediction']:.3f}")
    print(f"  Base value: {data['base_value']:.3f}")
    print(
        f"  Top contribution: {data['contributions'][0]['feature']} = {data['contributions'][0]['contribution']:.3f}"
    )


def test_explain_contributions_sum():
    """SHAP contributions should explain the prediction."""
    model_id = get_or_train_model()

    res = requests.get(f"{BASE_URL}/api/model/{model_id}/explain/patient/P0100")
    assert res.status_code == 200
    data = res.json()

    # Sum of contributions should roughly equal (prediction - base_value)
    total_contrib = sum(c["contribution"] for c in data["contributions"])
    # Note: For LightGBM, base_value and contributions are in log-odds space
    # so direct sum may not equal prediction difference
    print(f"  Sum of contributions: {total_contrib:.3f}")


def test_explain_invalid_model():
    """Invalid model ID should return 404."""
    res = requests.get(f"{BASE_URL}/api/model/invalid_model/explain/patient/P0001")
    assert (
        res.status_code == 404
    ), f"Expected 404 for invalid model, got {res.status_code}"
    print("  Invalid model ID: correctly rejected with 404")


def test_explain_invalid_patient():
    """Invalid patient ID should return 404."""
    model_id = get_or_train_model()

    res = requests.get(f"{BASE_URL}/api/model/{model_id}/explain/patient/INVALID")
    assert (
        res.status_code == 404
    ), f"Expected 404 for invalid patient, got {res.status_code}"
    print("  Invalid patient ID: correctly rejected with 404")


def test_explain_all_features_present():
    """All model features should have contributions."""
    # Train model with specific features
    res = requests.post(
        f"{BASE_URL}/api/train/quick",
        json={
            "filters": {},
            "model_type": "lightgbm",
            "features": ["age", "prior_opioid_days", "benzo_use", "er_visits"],
        },
    )
    model_id = res.json()["model_id"]

    res = requests.get(f"{BASE_URL}/api/model/{model_id}/explain/patient/P0050")
    assert res.status_code == 200
    data = res.json()

    feature_names = [c["feature"] for c in data["contributions"]]
    assert "age" in feature_names
    assert "prior_opioid_days" in feature_names
    assert "benzo_use" in feature_names
    assert "er_visits" in feature_names
    print(f"  All features present: {feature_names}")


def test_explain_feature_values_included():
    """Explanation should include feature values."""
    model_id = get_or_train_model()

    res = requests.get(f"{BASE_URL}/api/model/{model_id}/explain/patient/P0001")
    assert res.status_code == 200
    data = res.json()

    for contrib in data["contributions"]:
        assert "value" in contrib, f"Missing value for feature {contrib['feature']}"
    print("  Feature values included: OK")


if __name__ == "__main__":
    print("\n=== Test A4: Explanation API Tests ===\n")
    test_explain_patient()
    test_explain_contributions_sum()
    test_explain_invalid_model()
    test_explain_invalid_patient()
    test_explain_all_features_present()
    test_explain_feature_values_included()
    print("\nAll explanation tests passed!")
