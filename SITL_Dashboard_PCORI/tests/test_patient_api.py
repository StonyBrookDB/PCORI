"""
Test Patient API: Regression tests for /api/patients/{patient_id} endpoint.
Tests that patient predictions work with ALL model types.
"""

import requests

BASE_URL = "http://localhost:8501"


def test_patient_with_lightgbm_model():
    """After training lightgbm, get_patient should work."""
    # Train lightgbm model
    train_res = requests.post(
        f"{BASE_URL}/api/train/quick",
        json={
            "filters": {},
            "model_type": "lightgbm",
            "features": ["age", "prior_opioid_days", "benzo_use"],
        },
    )
    assert train_res.status_code == 200, f"Training failed: {train_res.text}"
    model_id = train_res.json()["model_id"]

    # Now get patient - this should NOT fail
    patient_res = requests.get(f"{BASE_URL}/api/patients/P001")
    assert (
        patient_res.status_code == 200
    ), f"Patient endpoint failed: {patient_res.text}"

    data = patient_res.json()
    assert "predictions" in data
    # Should have the lightgbm model prediction
    assert model_id in data["predictions"]
    pred = data["predictions"][model_id]
    assert pred["model_type"] == "lightgbm"
    assert 0 <= pred["probability"] <= 1
    print(f"  LightGBM prediction for P001: {pred['probability']:.3f}")


def test_patient_with_logreg_model():
    """After training logreg, get_patient should work."""
    # Train logreg model
    train_res = requests.post(
        f"{BASE_URL}/api/train/quick",
        json={
            "filters": {},
            "model_type": "logreg",
            "features": ["age", "prior_opioid_days"],
        },
    )
    assert train_res.status_code == 200, f"Training failed: {train_res.text}"
    model_id = train_res.json()["model_id"]

    # Now get patient - this should NOT fail (THIS WAS THE BUG!)
    patient_res = requests.get(f"{BASE_URL}/api/patients/P001")
    assert (
        patient_res.status_code == 200
    ), f"Patient endpoint failed: {patient_res.text}"

    data = patient_res.json()
    assert "predictions" in data
    # Should have the logreg model prediction
    logreg_preds = [
        p for mid, p in data["predictions"].items() if p["model_type"] == "logreg"
    ]
    assert len(logreg_preds) > 0, "No logreg predictions found"
    pred = logreg_preds[0]
    assert 0 <= pred["probability"] <= 1
    print(f"  LogReg prediction for P001: {pred['probability']:.3f}")


def test_patient_with_multiple_models():
    """Patient endpoint should return predictions from all trained models."""
    # Train both model types
    requests.post(
        f"{BASE_URL}/api/train/quick",
        json={
            "filters": {},
            "model_type": "lightgbm",
            "features": ["age", "prior_opioid_days", "benzo_use"],
        },
    )
    requests.post(
        f"{BASE_URL}/api/train/quick",
        json={"filters": {}, "model_type": "logreg", "features": ["age", "er_visits"]},
    )

    # Get patient
    patient_res = requests.get(f"{BASE_URL}/api/patients/P050")
    assert patient_res.status_code == 200

    data = patient_res.json()
    predictions = data["predictions"]

    # Should have multiple predictions
    assert len(predictions) >= 2, f"Expected at least 2 models, got {len(predictions)}"

    # Check we have both model types
    model_types = set(p["model_type"] for p in predictions.values())
    assert "lightgbm" in model_types, "Missing lightgbm predictions"
    assert "logreg" in model_types, "Missing logreg predictions"
    print(f"  Multiple models: {len(predictions)} predictions from {model_types}")


def test_patient_not_found():
    """Invalid patient ID should return 404."""
    res = requests.get(f"{BASE_URL}/api/patients/INVALID")
    assert res.status_code == 404, f"Expected 404, got {res.status_code}"
    print("  Invalid patient ID: correctly rejected with 404")


def test_patient_prediction_range():
    """Predictions should be valid probabilities between 0 and 1."""
    # Train a model
    requests.post(
        f"{BASE_URL}/api/train/quick",
        json={
            "filters": {},
            "model_type": "logreg",
            "features": ["age", "prior_opioid_days", "mental_health_score"],
        },
    )

    # Check multiple patients
    for pid in ["P001", "P050", "P100"]:
        res = requests.get(f"{BASE_URL}/api/patients/{pid}")
        assert res.status_code == 200
        data = res.json()
        for model_id, pred in data["predictions"].items():
            assert (
                0 <= pred["probability"] <= 1
            ), f"Invalid probability {pred['probability']} for {pid}"
            assert pred["prediction"] in [
                0,
                1,
            ], f"Invalid prediction {pred['prediction']} for {pid}"
    print("  All predictions in valid range [0, 1]")


if __name__ == "__main__":
    print("\n=== Test Patient API: Regression Tests ===\n")
    test_patient_with_lightgbm_model()
    test_patient_with_logreg_model()
    test_patient_with_multiple_models()
    test_patient_not_found()
    test_patient_prediction_range()
    print("\n" + "=" * 50)
    print("All patient API tests passed!")
    print("=" * 50)
