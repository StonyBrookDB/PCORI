"""
API Behavior Tests for SITL Dashboard

These tests capture the current behavior of the SITL dashboard API before
removing yl.* dependencies. They serve as regression tests to ensure the
API contract remains stable after refactoring.

Tests run against localhost:8503 and verify:
1. Health check endpoint
2. Patient list endpoint  
3. Cohort filter endpoint
4. Quick train endpoint (with patient_ids)
5. SHAP explanation endpoint

Run with: pytest tests/test_api_behavior.py -v
"""

import pytest
import requests
from typing import Optional

# Configuration
BASE_URL = "http://localhost:8503"
TIMEOUT = 30  # seconds


class TestHealthCheck:
    """Tests for GET /health endpoint."""
    
    def test_health_returns_200(self):
        """Health check should return 200 status code."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        
    def test_health_returns_status_ok(self):
        """Health check should return status: ok."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        data = response.json()
        assert data["status"] == "ok"
        
    def test_health_returns_version(self):
        """Health check should include version string."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)
        
    def test_health_returns_patient_count(self):
        """Health check should include patient count."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        data = response.json()
        assert "patient_count" in data
        assert isinstance(data["patient_count"], int)
        assert data["patient_count"] > 0
        
    def test_health_returns_models_loaded(self):
        """Health check should include models loaded count."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        data = response.json()
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], int)


class TestPatientList:
    """Tests for GET /api/patients endpoint."""
    
    def test_patients_returns_200(self):
        """Patient list should return 200 status code."""
        response = requests.get(f"{BASE_URL}/api/patients", timeout=TIMEOUT)
        assert response.status_code == 200
        
    def test_patients_returns_list(self):
        """Patient list should return a patients array."""
        response = requests.get(f"{BASE_URL}/api/patients", timeout=TIMEOUT)
        data = response.json()
        assert "patients" in data
        assert isinstance(data["patients"], list)
        
    def test_patients_list_not_empty(self):
        """Patient list should not be empty."""
        response = requests.get(f"{BASE_URL}/api/patients", timeout=TIMEOUT)
        data = response.json()
        assert len(data["patients"]) > 0
        
    def test_patient_has_required_fields(self):
        """Each patient should have id, name, split, and outcome fields."""
        response = requests.get(f"{BASE_URL}/api/patients", timeout=TIMEOUT)
        data = response.json()
        patient = data["patients"][0]
        
        assert "id" in patient
        assert "name" in patient
        assert "split" in patient
        assert "outcome" in patient
        
    def test_patient_id_format(self):
        """Patient IDs should follow P### format."""
        response = requests.get(f"{BASE_URL}/api/patients", timeout=TIMEOUT)
        data = response.json()
        patient = data["patients"][0]
        
        assert patient["id"].startswith("P")
        assert len(patient["id"]) >= 2
        
    def test_patient_split_values(self):
        """Patient split should be train, validation, or test."""
        response = requests.get(f"{BASE_URL}/api/patients", timeout=TIMEOUT)
        data = response.json()
        
        valid_splits = {"train", "validation", "test"}
        for patient in data["patients"][:10]:  # Check first 10
            assert patient["split"] in valid_splits, f"Invalid split: {patient[split]}"
            
    def test_patient_outcome_values(self):
        """Patient outcome should be 0 or 1."""
        response = requests.get(f"{BASE_URL}/api/patients", timeout=TIMEOUT)
        data = response.json()
        
        for patient in data["patients"][:10]:  # Check first 10
            assert patient["outcome"] in [0, 1], f"Invalid outcome: {patient[outcome]}"


class TestCohortFilter:
    """Tests for POST /api/cohort/filter endpoint."""
    
    def test_filter_returns_200(self):
        """Cohort filter should return 200 status code."""
        response = requests.post(
            f"{BASE_URL}/api/cohort/filter",
            json={},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        
    def test_filter_returns_patient_ids(self):
        """Cohort filter should return patient_ids array."""
        response = requests.post(
            f"{BASE_URL}/api/cohort/filter",
            json={},
            timeout=TIMEOUT
        )
        data = response.json()
        assert "patient_ids" in data
        assert isinstance(data["patient_ids"], list)
        
    def test_filter_empty_returns_all(self):
        """Empty filter criteria should return all patients."""
        response = requests.post(
            f"{BASE_URL}/api/cohort/filter",
            json={},
            timeout=TIMEOUT
        )
        data = response.json()
        # Should have substantial number of patients
        assert len(data["patient_ids"]) >= 1000
        
    def test_filter_patient_id_format(self):
        """Filtered patient IDs should follow P#### format."""
        response = requests.post(
            f"{BASE_URL}/api/cohort/filter",
            json={},
            timeout=TIMEOUT
        )
        data = response.json()
        
        for pid in data["patient_ids"][:10]:  # Check first 10
            assert pid.startswith("P")
            
    def test_filter_with_age_criteria(self):
        """Filter with age criteria should reduce result set."""
        # Get all patients first
        all_response = requests.post(
            f"{BASE_URL}/api/cohort/filter",
            json={},
            timeout=TIMEOUT
        )
        all_count = len(all_response.json()["patient_ids"])
        
        # Filter with age restriction
        filtered_response = requests.post(
            f"{BASE_URL}/api/cohort/filter",
            json={"age_min": 50, "age_max": 60},
            timeout=TIMEOUT
        )
        filtered_count = len(filtered_response.json()["patient_ids"])
        
        # Filtered should be less than all
        assert filtered_count < all_count
        assert filtered_count > 0  # But not empty


class TestQuickTrain:
    """Tests for POST /api/train/quick endpoint."""
    
    @pytest.fixture
    def patient_ids(self):
        """Get a sample of patient IDs for training."""
        response = requests.post(
            f"{BASE_URL}/api/cohort/filter",
            json={},
            timeout=TIMEOUT
        )
        all_ids = response.json()["patient_ids"]
        # Return first 200 for faster training
        return all_ids[:200]
    
    def test_quick_train_returns_200(self, patient_ids):
        """Quick train should return 200 status code."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days", "benzo_use"]
            },
            timeout=60  # Training may take longer
        )
        assert response.status_code == 200
        
    def test_quick_train_returns_model_id(self, patient_ids):
        """Quick train should return a model_id."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days", "benzo_use"]
            },
            timeout=60
        )
        data = response.json()
        assert "model_id" in data
        assert isinstance(data["model_id"], str)
        assert len(data["model_id"]) > 0
        
    def test_quick_train_returns_metrics(self, patient_ids):
        """Quick train should return metrics including auroc."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days", "benzo_use"]
            },
            timeout=60
        )
        data = response.json()
        
        assert "metrics" in data
        assert isinstance(data["metrics"], dict)
        assert "auroc" in data["metrics"]
        
        # AUROC should be between 0 and 1
        auroc = data["metrics"]["auroc"]
        assert 0.0 <= auroc <= 1.0
        
    def test_quick_train_returns_feature_importance(self, patient_ids):
        """Quick train should return feature_importance dict."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days", "benzo_use"]
            },
            timeout=60
        )
        data = response.json()
        
        assert "feature_importance" in data
        assert isinstance(data["feature_importance"], list)
        assert len(data["feature_importance"]) == 3  # Same as number of features
        
        # Each feature importance entry should have feature and importance
        for item in data["feature_importance"]:
            assert "feature" in item
            assert "importance" in item
            assert isinstance(item["importance"], (int, float))
            
    def test_quick_train_feature_importance_contains_all_features(self, patient_ids):
        """Feature importance should include all training features."""
        features = ["age", "prior_opioid_days", "benzo_use"]
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": features
            },
            timeout=60
        )
        data = response.json()
        
        importance_features = {item["feature"] for item in data["feature_importance"]}
        for feature in features:
            assert feature in importance_features
            
    def test_quick_train_returns_model_type(self, patient_ids):
        """Quick train should return the model_type."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days"]
            },
            timeout=60
        )
        data = response.json()
        
        assert "model_type" in data
        assert data["model_type"] == "lightgbm"
        
    def test_quick_train_logreg(self, patient_ids):
        """Quick train should work with logreg model type."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "logreg",
                "features": ["age", "prior_opioid_days"]
            },
            timeout=60
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "logreg"
        assert "model_id" in data
        assert "metrics" in data
        

class TestShapExplanation:
    """Tests for GET /api/model/{id}/explain/patient/{pid} endpoint."""
    
    @pytest.fixture
    def trained_model(self):
        """Train a model and return its ID."""
        # Get patient IDs first
        cohort_response = requests.post(
            f"{BASE_URL}/api/cohort/filter",
            json={},
            timeout=TIMEOUT
        )
        patient_ids = cohort_response.json()["patient_ids"][:200]
        
        # Train model
        train_response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days", "benzo_use"]
            },
            timeout=60
        )
        return train_response.json()["model_id"]
    
    def test_explain_returns_200(self, trained_model):
        """SHAP explanation should return 200 status code."""
        response = requests.get(
            f"{BASE_URL}/api/model/{trained_model}/explain/patient/P0001",
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        
    def test_explain_returns_patient_id(self, trained_model):
        """SHAP explanation should return the patient_id."""
        response = requests.get(
            f"{BASE_URL}/api/model/{trained_model}/explain/patient/P0001",
            timeout=TIMEOUT
        )
        data = response.json()
        assert "patient_id" in data
        assert data["patient_id"] == "P0001"
        
    def test_explain_returns_model_id(self, trained_model):
        """SHAP explanation should return the model_id."""
        response = requests.get(
            f"{BASE_URL}/api/model/{trained_model}/explain/patient/P0001",
            timeout=TIMEOUT
        )
        data = response.json()
        assert "model_id" in data
        assert data["model_id"] == trained_model
        
    def test_explain_returns_prediction(self, trained_model):
        """SHAP explanation should return prediction value."""
        response = requests.get(
            f"{BASE_URL}/api/model/{trained_model}/explain/patient/P0001",
            timeout=TIMEOUT
        )
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], (int, float))
        # Prediction should be a probability between 0 and 1
        assert 0.0 <= data["prediction"] <= 1.0
        
    def test_explain_returns_base_value(self, trained_model):
        """SHAP explanation should return base_value."""
        response = requests.get(
            f"{BASE_URL}/api/model/{trained_model}/explain/patient/P0001",
            timeout=TIMEOUT
        )
        data = response.json()
        assert "base_value" in data
        assert isinstance(data["base_value"], (int, float))
        
    def test_explain_returns_contributions(self, trained_model):
        """SHAP explanation should return contributions list."""
        response = requests.get(
            f"{BASE_URL}/api/model/{trained_model}/explain/patient/P0001",
            timeout=TIMEOUT
        )
        data = response.json()
        assert "contributions" in data
        assert isinstance(data["contributions"], list)
        assert len(data["contributions"]) > 0
        
    def test_explain_contribution_structure(self, trained_model):
        """Each contribution should have feature, value, and contribution."""
        response = requests.get(
            f"{BASE_URL}/api/model/{trained_model}/explain/patient/P0001",
            timeout=TIMEOUT
        )
        data = response.json()
        
        for contrib in data["contributions"]:
            assert "feature" in contrib
            assert "value" in contrib
            assert "contribution" in contrib
            assert isinstance(contrib["feature"], str)
            assert isinstance(contrib["contribution"], (int, float))
            
    def test_explain_invalid_model_returns_404(self):
        """Invalid model ID should return 404."""
        response = requests.get(
            f"{BASE_URL}/api/model/nonexistent_model/explain/patient/P0001",
            timeout=TIMEOUT
        )
        assert response.status_code == 404
        
    def test_explain_invalid_patient_returns_404(self, trained_model):
        """Invalid patient ID should return 404."""
        response = requests.get(
            f"{BASE_URL}/api/model/{trained_model}/explain/patient/INVALID_PATIENT",
            timeout=TIMEOUT
        )
        assert response.status_code == 404


class TestMetricsStructure:
    """Tests to capture the exact structure of metrics returned by training."""
    
    @pytest.fixture
    def patient_ids(self):
        """Get a sample of patient IDs for training."""
        response = requests.post(
            f"{BASE_URL}/api/cohort/filter",
            json={},
            timeout=TIMEOUT
        )
        return response.json()["patient_ids"][:200]
    
    def test_metrics_contains_auroc(self, patient_ids):
        """Metrics should contain auroc."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days"]
            },
            timeout=60
        )
        metrics = response.json()["metrics"]
        assert "auroc" in metrics
        
    def test_metrics_contains_auprc(self, patient_ids):
        """Metrics should contain auprc (area under precision-recall curve)."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days"]
            },
            timeout=60
        )
        metrics = response.json()["metrics"]
        assert "auprc" in metrics
        
    def test_metrics_contains_accuracy(self, patient_ids):
        """Metrics should contain accuracy."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days"]
            },
            timeout=60
        )
        metrics = response.json()["metrics"]
        assert "accuracy" in metrics
        
    def test_all_metrics_are_valid_numbers(self, patient_ids):
        """All metrics should be valid numbers between 0 and 1."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days"]
            },
            timeout=60
        )
        metrics = response.json()["metrics"]
        
        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"{key} is not a number"
            assert 0.0 <= value <= 1.0, f"{key}={value} not in [0,1]"


class TestTrainingResponseStructure:
    """Tests to capture the full training response structure."""
    
    @pytest.fixture
    def patient_ids(self):
        """Get a sample of patient IDs for training."""
        response = requests.post(
            f"{BASE_URL}/api/cohort/filter",
            json={},
            timeout=TIMEOUT
        )
        return response.json()["patient_ids"][:200]
    
    def test_response_contains_cohort_size(self, patient_ids):
        """Training response should include cohort_size."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days"]
            },
            timeout=60
        )
        data = response.json()
        assert "cohort_size" in data
        assert isinstance(data["cohort_size"], int)
        
    def test_response_contains_train_size(self, patient_ids):
        """Training response should include train_size."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days"]
            },
            timeout=60
        )
        data = response.json()
        assert "train_size" in data
        assert isinstance(data["train_size"], int)
        
    def test_response_contains_test_size(self, patient_ids):
        """Training response should include test_size."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days"]
            },
            timeout=60
        )
        data = response.json()
        assert "test_size" in data
        assert isinstance(data["test_size"], int)
        
    def test_response_contains_training_time(self, patient_ids):
        """Training response should include training_time_seconds."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days"]
            },
            timeout=60
        )
        data = response.json()
        assert "training_time_seconds" in data
        assert isinstance(data["training_time_seconds"], (int, float))
        assert data["training_time_seconds"] >= 0
        
    def test_response_contains_warnings(self, patient_ids):
        """Training response should include warnings list."""
        response = requests.post(
            f"{BASE_URL}/api/train/quick",
            json={
                "patient_ids": patient_ids,
                "model_type": "lightgbm",
                "features": ["age", "prior_opioid_days"]
            },
            timeout=60
        )
        data = response.json()
        assert "warnings" in data
        assert isinstance(data["warnings"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
