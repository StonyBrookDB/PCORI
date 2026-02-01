"""
Test E2E: End-to-End Workflow Test
Tests the complete workflow: Filter -> Train -> Explain
"""

import time

import requests

BASE_URL = "http://localhost:8501"


def test_full_workflow():
    """Filter -> Train -> Explain must all succeed."""
    print("\n  Step 1: Apply cohort filter")
    # 1. Filter cohort
    filter_res = requests.post(
        f"{BASE_URL}/api/cohort/filter",
        json={"age_min": 40, "age_max": 65, "cancer_status": False},
    )
    assert filter_res.status_code == 200, f"Filter failed: {filter_res.text}"
    cohort = filter_res.json()
    print(
        f"    Cohort: {cohort['total_count']} patients, {cohort['event_count']} events"
    )

    assert cohort["total_count"] >= 100, "Cohort too small for training"

    print("\n  Step 2: Train model on filtered cohort")
    # 2. Train model
    train_start = time.time()
    train_res = requests.post(
        f"{BASE_URL}/api/train/quick",
        json={
            "filters": {"age_min": 40, "age_max": 65, "cancer_status": False},
            "model_type": "lightgbm",
            "features": [
                "age",
                "prior_opioid_days",
                "benzo_use",
                "mental_health_score",
            ],
        },
    )
    train_time = time.time() - train_start
    assert train_res.status_code == 200, f"Training failed: {train_res.text}"
    model = train_res.json()
    print(f"    Model ID: {model['model_id']}")
    print(f"    AUROC: {model['metrics']['auroc']:.3f}")
    print(f"    Training time: {train_time:.2f}s")

    assert model["metrics"]["auroc"] > 0.5, "Model not better than random"

    print("\n  Step 3: Get explanation for a patient")
    # 3. Explain patient
    patient_id = cohort["patient_ids"][0]
    explain_res = requests.get(
        f"{BASE_URL}/api/model/{model['model_id']}/explain/patient/{patient_id}"
    )
    assert explain_res.status_code == 200, f"Explanation failed: {explain_res.text}"
    explanation = explain_res.json()
    print(f"    Patient: {patient_id}")
    print(f"    Prediction: {explanation['prediction']:.3f}")
    print(f"    Top factors:")
    for c in explanation["contributions"][:3]:
        sign = "+" if c["contribution"] > 0 else ""
        print(
            f"      {c['feature']}: {sign}{c['contribution']:.3f} (value: {c['value']})"
        )

    print("\n  Full workflow completed successfully!")


def test_health_endpoint():
    """Health check should return status and counts."""
    res = requests.get(f"{BASE_URL}/health")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "ok"
    assert data["patient_count"] >= 2000
    print(
        f"  Health: status={data['status']}, patients={data['patient_count']}, models={data['models_loaded']}"
    )


def test_models_persist_in_memory():
    """Trained models should be available via /api/models."""
    # Train a model
    requests.post(
        f"{BASE_URL}/api/train/quick",
        json={
            "filters": {},
            "model_type": "logreg",
            "features": ["age", "prior_opioid_days"],
        },
    )

    # Check it appears in models list
    res = requests.get(f"{BASE_URL}/api/models")
    assert res.status_code == 200
    models = res.json()

    assert len(models) > 0, "No models found after training"
    print(f"  Models in memory: {len(models)}")

    # Verify model details
    model_id = list(models.keys())[0]
    assert "val_metrics" in models[model_id]
    assert "features" in models[model_id]


if __name__ == "__main__":
    print("\n=== Test E2E: End-to-End Workflow Tests ===\n")
    test_health_endpoint()
    test_full_workflow()
    test_models_persist_in_memory()
    print("\n" + "=" * 50)
    print("All end-to-end tests passed!")
    print("=" * 50)
