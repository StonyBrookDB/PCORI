"""
Test A3: Training API Tests
Tests for the /api/train/quick endpoint.
"""

import time

import requests

BASE_URL = "http://localhost:8501"


def test_train_lightgbm():
    """LightGBM training should succeed with valid cohort."""
    res = requests.post(
        f"{BASE_URL}/api/train/quick",
        json={
            "filters": {"age_min": 30, "age_max": 70},
            "model_type": "lightgbm",
            "features": [
                "age",
                "prior_opioid_days",
                "benzo_use",
                "mental_health_score",
            ],
        },
    )
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()

    assert "model_id" in data, "Response missing model_id"
    assert "metrics" in data, "Response missing metrics"
    assert (
        data["metrics"]["auroc"] > 0.5
    ), f"AUROC {data['metrics']['auroc']} not > 0.5 (random)"
    print(
        f"  LightGBM: AUROC={data['metrics']['auroc']:.3f}, time={data['training_time_seconds']:.2f}s"
    )


def test_train_logreg():
    """Logistic regression training should succeed."""
    res = requests.post(
        f"{BASE_URL}/api/train/quick",
        json={
            "filters": {},
            "model_type": "logreg",
            "features": ["age", "prior_opioid_days", "benzo_use"],
        },
    )
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()

    assert data["metrics"]["auroc"] > 0.5
    print(
        f"  LogReg: AUROC={data['metrics']['auroc']:.3f}, time={data['training_time_seconds']:.2f}s"
    )


def test_train_small_cohort_rejected():
    """Cohort < 100 must return 400."""
    res = requests.post(
        f"{BASE_URL}/api/train/quick",
        json={
            "filters": {"age_min": 95, "age_max": 100},
            "model_type": "lightgbm",
            "features": ["age"],
        },
    )
    assert (
        res.status_code == 400
    ), f"Expected 400 for small cohort, got {res.status_code}"
    print("  Small cohort: correctly rejected with 400")


def test_train_invalid_features_rejected():
    """Invalid features must return 422."""
    res = requests.post(
        f"{BASE_URL}/api/train/quick",
        json={"filters": {}, "model_type": "lightgbm", "features": ["invalid_feature"]},
    )
    assert (
        res.status_code == 422
    ), f"Expected 422 for invalid features, got {res.status_code}"
    print("  Invalid features: correctly rejected with 422")


def test_train_empty_features_rejected():
    """Empty features list must return 422."""
    res = requests.post(
        f"{BASE_URL}/api/train/quick",
        json={"filters": {}, "model_type": "lightgbm", "features": []},
    )
    assert (
        res.status_code == 422
    ), f"Expected 422 for empty features, got {res.status_code}"
    print("  Empty features: correctly rejected with 422")


def test_train_invalid_model_type_rejected():
    """Invalid model type must return 422."""
    res = requests.post(
        f"{BASE_URL}/api/train/quick",
        json={"filters": {}, "model_type": "invalid_model", "features": ["age"]},
    )
    assert (
        res.status_code == 422
    ), f"Expected 422 for invalid model type, got {res.status_code}"
    print("  Invalid model type: correctly rejected with 422")


def test_train_performance():
    """Training must complete in < 20 seconds."""
    start = time.time()
    res = requests.post(
        f"{BASE_URL}/api/train/quick",
        json={
            "filters": {},
            "model_type": "lightgbm",
            "features": [
                "age",
                "prior_opioid_days",
                "benzo_use",
                "mental_health_score",
                "er_visits",
            ],
        },
    )
    elapsed = time.time() - start
    assert res.status_code == 200
    assert elapsed < 20.0, f"Training took {elapsed:.1f}s, expected < 20s"
    print(f"  Performance: {elapsed:.2f}s")


def test_train_returns_feature_importance():
    """Training should return feature importance."""
    res = requests.post(
        f"{BASE_URL}/api/train/quick",
        json={
            "filters": {},
            "model_type": "lightgbm",
            "features": ["age", "prior_opioid_days", "benzo_use"],
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert "feature_importance" in data, "Response missing feature_importance"
    assert (
        len(data["feature_importance"]) == 3
    ), f"Expected 3 features, got {len(data['feature_importance'])}"
    print(f"  Feature importance: {[f['feature'] for f in data['feature_importance']]}")


if __name__ == "__main__":
    print("\n=== Test A3: Training API Tests ===\n")
    test_train_lightgbm()
    test_train_logreg()
    test_train_small_cohort_rejected()
    test_train_invalid_features_rejected()
    test_train_empty_features_rejected()
    test_train_invalid_model_type_rejected()
    test_train_performance()
    test_train_returns_feature_importance()
    print("\nAll training tests passed!")
