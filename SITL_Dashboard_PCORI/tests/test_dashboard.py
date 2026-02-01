"""
Automated tests for SITL Dashboard.

Run with:
    cd /home/yinan/checkouts/v0/apps/sitl_dashboard
    PYTHONPATH="/home/yinan/checkouts/v0/packages" \
    conda run -n eliu3 python tests/test_dashboard.py

Or with pytest:
    PYTHONPATH="/home/yinan/checkouts/v0/packages" \
    conda run -n eliu3 python -m pytest tests/test_dashboard.py -v
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import TestCase, main

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test database path
TEST_DB_DIR = tempfile.mkdtemp()
os.environ["SITL_TEST_MODE"] = "1"


class TestDatabaseOperations(TestCase):
    """Test database CRUD operations."""

    def setUp(self):
        """Set up test database."""
        from backend.job_integration import DASHBOARD_DB_PATH, init_dashboard_db

        # Use a test-specific database
        self.original_db_path = DASHBOARD_DB_PATH
        import backend.job_integration as ji

        ji.DASHBOARD_DB_PATH = Path(TEST_DB_DIR) / "test_jobs.db"
        init_dashboard_db()

    def test_init_creates_tables(self):
        """Test that init creates jobs and models tables."""
        import sqlite3

        from backend.job_integration import DASHBOARD_DB_PATH

        conn = sqlite3.connect(DASHBOARD_DB_PATH)
        cursor = conn.cursor()

        # Check jobs table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
        )
        self.assertIsNotNone(cursor.fetchone(), "jobs table should exist")

        # Check models table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='models'"
        )
        self.assertIsNotNone(cursor.fetchone(), "models table should exist")

        conn.close()

    def test_save_and_get_job(self):
        """Test saving and retrieving a job."""
        from backend.job_integration import DashboardJob, get_job, save_job

        job = DashboardJob(
            job_id="test_job_001",
            session_id="test_session",
            dataset_id="synthetic",
            model_type="lightgbm",
            model_name="test_model",
            features=["age", "bmi"],
            config={"n_estimators": 100},
            status="queued",
            created_at="2024-12-28T10:00:00",
            work_dir="/tmp/test",
            log_path="/tmp/test/log.txt",
            metrics_path="/tmp/test/metrics.json",
            error=None,
        )
        save_job(job)

        retrieved = get_job("test_job_001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.job_id, "test_job_001")
        self.assertEqual(retrieved.dataset_id, "synthetic")
        self.assertEqual(retrieved.model_type, "lightgbm")
        self.assertEqual(retrieved.features, ["age", "bmi"])

    def test_save_and_get_model(self):
        """Test saving and retrieving a model."""
        from backend.job_integration import TrainedModel, get_model, save_model

        model = TrainedModel(
            model_id="test_model_001",
            job_id="test_job_001",
            dataset_id="synthetic",
            model_type="lightgbm",
            model_name="My Test Model",
            task_type="classification",
            features=["age", "bmi"],
            train_metrics={"auc": 0.85},
            val_metrics={"auc": 0.82},
            feature_importance=[{"feature": "age", "importance": 0.6}],
            training_history={"train": [{"epoch": 0, "loss": 0.5}]},
            model_path="/tmp/model.pkl",
            work_dir="/tmp/test",
            created_at="2024-12-28T10:00:00",
        )
        save_model(model)

        retrieved = get_model("test_model_001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.model_id, "test_model_001")
        self.assertEqual(retrieved.dataset_id, "synthetic")
        self.assertEqual(retrieved.model_name, "My Test Model")
        self.assertEqual(retrieved.train_metrics, {"auc": 0.85})

    def test_list_models_filter_by_dataset(self):
        """Test filtering models by dataset."""
        from backend.job_integration import TrainedModel, list_models, save_model

        # Save models for different datasets
        for i, ds in enumerate(["synthetic", "synthetic", "breast_cancer"]):
            model = TrainedModel(
                model_id=f"filter_test_{i}",
                job_id=f"job_{i}",
                dataset_id=ds,
                model_type="lightgbm",
                model_name=f"Model {i}",
                task_type="classification",
                features=["a"],
                train_metrics={},
                val_metrics={},
                feature_importance=[],
                training_history={},
                model_path="",
                work_dir="",
                created_at=f"2024-12-28T10:0{i}:00",
            )
            save_model(model)

        # Filter by synthetic
        synthetic_models = list_models(dataset_id="synthetic")
        synthetic_ids = [m.model_id for m in synthetic_models]
        self.assertIn("filter_test_0", synthetic_ids)
        self.assertIn("filter_test_1", synthetic_ids)
        self.assertNotIn("filter_test_2", synthetic_ids)

        # Filter by breast_cancer
        bc_models = list_models(dataset_id="breast_cancer")
        bc_ids = [m.model_id for m in bc_models]
        self.assertIn("filter_test_2", bc_ids)
        self.assertNotIn("filter_test_0", bc_ids)

    def test_list_models_search(self):
        """Test searching models by name."""
        from backend.job_integration import TrainedModel, list_models, save_model

        # Save models with different names
        for name in ["Alpha Model", "Beta Model", "Gamma"]:
            model = TrainedModel(
                model_id=f"search_{name.replace(' ', '_').lower()}",
                job_id="job",
                dataset_id="synthetic",
                model_type="lightgbm",
                model_name=name,
                task_type="classification",
                features=["a"],
                train_metrics={},
                val_metrics={},
                feature_importance=[],
                training_history={},
                model_path="",
                work_dir="",
                created_at="2024-12-28T10:00:00",
            )
            save_model(model)

        # Search for "Model"
        results = list_models(search_query="Model")
        names = [m.model_name for m in results]
        self.assertIn("Alpha Model", names)
        self.assertIn("Beta Model", names)
        self.assertNotIn("Gamma", names)

    def test_model_to_dict(self):
        """Test converting model to dictionary."""
        from backend.job_integration import TrainedModel, model_to_dict

        model = TrainedModel(
            model_id="dict_test",
            job_id="job",
            dataset_id="synthetic",
            model_type="pytorch",
            model_name="Dict Test Model",
            task_type="classification",
            features=["x", "y"],
            train_metrics={"loss": 0.1},
            val_metrics={"loss": 0.2},
            feature_importance=[],
            training_history={},
            model_path="/path/to/model",
            work_dir="/tmp",
            created_at="2024-12-28T10:00:00",
        )

        d = model_to_dict(model)
        self.assertEqual(d["model_id"], "dict_test")
        self.assertEqual(d["name"], "Dict Test Model")
        self.assertEqual(d["dataset_id"], "synthetic")
        self.assertEqual(d["train_metrics"], {"loss": 0.1})


class TestTrainingRunner(TestCase):
    """Test training runner functionality."""

    def test_lightgbm_training(self):
        """Test LightGBM training completes."""
        import numpy as np
        from backend.training_runner import TrainingRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = TrainingRunner(
                model_type="lightgbm",
                model_id="test_lgb",
                work_dir=Path(tmpdir),
                task_type="classification",
            )

            # Create dummy data
            np.random.seed(42)
            X_train = np.random.randn(100, 5)
            y_train = (np.random.randn(100) > 0).astype(int)
            X_val = np.random.randn(20, 5)
            y_val = (np.random.randn(20) > 0).astype(int)

            result = runner.run(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                config={"n_estimators": 10, "max_depth": 3},
                feature_names=["f1", "f2", "f3", "f4", "f5"],
            )

            self.assertTrue(result.success, f"Training failed: {result.error}")
            self.assertIn("val_auc", result.metrics)

    def test_logreg_training(self):
        """Test Logistic Regression training completes."""
        import numpy as np
        from backend.training_runner import TrainingRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = TrainingRunner(
                model_type="logreg",
                model_id="test_logreg",
                work_dir=Path(tmpdir),
                task_type="classification",
            )

            np.random.seed(42)
            X_train = np.random.randn(100, 5)
            y_train = (np.random.randn(100) > 0).astype(int)
            X_val = np.random.randn(20, 5)
            y_val = (np.random.randn(20) > 0).astype(int)

            result = runner.run(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                config={"C": 1.0, "max_iter": 100},
                feature_names=["f1", "f2", "f3", "f4", "f5"],
            )

            self.assertTrue(result.success, f"Training failed: {result.error}")
            self.assertIn("val_auc", result.metrics)

    def test_pytorch_training(self):
        """Test PyTorch MLP training completes."""
        import numpy as np
        from backend.training_runner import TrainingRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = TrainingRunner(
                model_type="pytorch",
                model_id="test_pytorch",
                work_dir=Path(tmpdir),
                task_type="classification",
            )

            np.random.seed(42)
            X_train = np.random.randn(100, 5).astype(np.float32)
            y_train = (np.random.randn(100) > 0).astype(int)
            X_val = np.random.randn(20, 5).astype(np.float32)
            y_val = (np.random.randn(20) > 0).astype(int)

            result = runner.run(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                config={"hidden_layers": [16], "epochs": 2, "learning_rate": 0.01},
                feature_names=["f1", "f2", "f3", "f4", "f5"],
            )

            self.assertTrue(result.success, f"Training failed: {result.error}")


class TestAPIEndpoints(TestCase):
    """Test API endpoints using TestClient."""

    @classmethod
    def setUpClass(cls):
        """Set up test client."""
        try:
            from backend.main import app
            from fastapi.testclient import TestClient

            cls.client = TestClient(app)
            cls.has_client = True
        except ImportError:
            cls.has_client = False

    def test_health_endpoint(self):
        """Test that server responds."""
        if not self.has_client:
            self.skipTest("TestClient not available")

        # Just check the app can be imported
        from backend.main import app

        self.assertIsNotNone(app)

    def test_datasets_endpoint(self):
        """Test /api/datasets returns datasets."""
        if not self.has_client:
            self.skipTest("TestClient not available")

        response = self.client.get("/api/datasets")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("datasets", data)
        dataset_ids = [d["dataset_id"] for d in data["datasets"]]
        self.assertIn("synthetic", dataset_ids)

    def test_models_endpoint(self):
        """Test /api/models returns models."""
        if not self.has_client:
            self.skipTest("TestClient not available")

        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict)

    def test_jobs_endpoint(self):
        """Test /api/jobs returns jobs."""
        if not self.has_client:
            self.skipTest("TestClient not available")

        response = self.client.get("/api/jobs")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("jobs", data)


def run_tests():
    """Run all tests and print summary."""
    import unittest

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingRunner))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIEndpoints))

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFailed tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\nError tests:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
