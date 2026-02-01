"""
Test A1: Database Tests
Tests for database integrity and patient data quality.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import database as db
from backend.config import MIN_COHORT_SIZE


def test_patient_count():
    """DB must have >= 2000 patients."""
    count = db.get_patient_count()
    assert count >= 2000, f"Expected >= 2000 patients, got {count}"
    print(f"  Patient count: {count}")


def test_outcome_distribution():
    """Outcome rate must be 10-30%."""
    result = db.filter_patients()
    rate = result["event_rate"]
    assert 0.10 <= rate <= 0.30, f"Event rate {rate:.1%} outside valid range [10%, 30%]"
    print(f"  Event rate: {rate:.1%}")


def test_patient_retrieval():
    """Can retrieve individual patient by ID."""
    patient = db.get_patient("P0001")
    assert patient is not None, "Patient P0001 not found"
    assert "patient_id" in patient
    assert "age" in patient
    assert "outcome" in patient
    print(f"  Patient P0001: age={patient['age']}, outcome={patient['outcome']}")


def test_train_val_split():
    """Train/val split should be approximately 80/20."""
    result = db.filter_patients()
    train_ratio = result["train_count"] / result["total_count"]
    assert (
        0.75 <= train_ratio <= 0.85
    ), f"Train ratio {train_ratio:.1%} not in [75%, 85%]"
    print(
        f"  Train/Val: {result['train_count']}/{result['val_count']} ({train_ratio:.1%})"
    )


def test_feature_ranges():
    """Patient features should be within valid ranges."""
    patients = db.get_patients_by_ids(["P0001", "P0100", "P0500"])
    for p in patients:
        assert 18 <= p["age"] <= 100, f"Invalid age: {p['age']}"
        assert (
            0 <= p["mental_health_score"] <= 10
        ), f"Invalid mental health score: {p['mental_health_score']}"
        assert p["benzo_use"] in [0, 1], f"Invalid benzo_use: {p['benzo_use']}"
        assert p["cancer_status"] in [
            0,
            1,
        ], f"Invalid cancer_status: {p['cancer_status']}"
    print("  Feature ranges: OK")


if __name__ == "__main__":
    print("\n=== Test A1: Database Tests ===\n")
    test_patient_count()
    test_outcome_distribution()
    test_patient_retrieval()
    test_train_val_split()
    test_feature_ranges()
    print("\nAll database tests passed!")
