"""
Test A2: Cohort Filter API Tests
Tests for the /api/cohort/filter endpoint.
"""

import time

import requests

BASE_URL = "http://localhost:8501"


def test_filter_no_criteria():
    """Empty filter returns all patients."""
    res = requests.post(f"{BASE_URL}/api/cohort/filter", json={})
    assert res.status_code == 200
    data = res.json()
    assert (
        data["total_count"] >= 2000
    ), f"Expected >= 2000 patients, got {data['total_count']}"
    print(f"  All patients: {data['total_count']}")


def test_filter_reduces_cohort():
    """Restrictive filters must reduce cohort size."""
    # Broad filter
    broad = requests.post(f"{BASE_URL}/api/cohort/filter", json={}).json()

    # Narrow filter
    narrow = requests.post(
        f"{BASE_URL}/api/cohort/filter", json={"age_min": 50, "age_max": 55}
    ).json()

    assert (
        narrow["total_count"] < broad["total_count"]
    ), f"Narrow ({narrow['total_count']}) should be < broad ({broad['total_count']})"
    print(f"  Broad: {broad['total_count']}, Narrow: {narrow['total_count']}")


def test_filter_age_range():
    """Age filter returns patients within range."""
    res = requests.post(
        f"{BASE_URL}/api/cohort/filter", json={"age_min": 40, "age_max": 50}
    )
    assert res.status_code == 200
    data = res.json()
    assert (
        data["statistics"]["age_min"] >= 40
    ), f"Min age {data['statistics']['age_min']} < 40"
    assert (
        data["statistics"]["age_max"] <= 50
    ), f"Max age {data['statistics']['age_max']} > 50"
    print(
        f"  Age filter: {data['total_count']} patients, range {data['statistics']['age_min']}-{data['statistics']['age_max']}"
    )


def test_invalid_age_range_rejected():
    """age_min > age_max must return 422."""
    res = requests.post(
        f"{BASE_URL}/api/cohort/filter", json={"age_min": 70, "age_max": 30}
    )
    assert res.status_code == 422, f"Expected 422, got {res.status_code}"
    print("  Invalid age range: correctly rejected with 422")


def test_filter_cancer_status():
    """Cancer status filter works correctly."""
    cancer = requests.post(
        f"{BASE_URL}/api/cohort/filter", json={"cancer_status": True}
    ).json()
    no_cancer = requests.post(
        f"{BASE_URL}/api/cohort/filter", json={"cancer_status": False}
    ).json()

    assert (
        cancer["total_count"] < no_cancer["total_count"]
    ), "Cancer patients should be fewer than non-cancer"
    print(f"  Cancer: {cancer['total_count']}, No cancer: {no_cancer['total_count']}")


def test_filter_returns_warnings():
    """Small cohorts should return warnings."""
    res = requests.post(
        f"{BASE_URL}/api/cohort/filter", json={"age_min": 90, "age_max": 100}
    )
    assert res.status_code == 200
    data = res.json()
    # Very restrictive filter should produce warnings
    if data["total_count"] < 100:
        assert len(data["warnings"]) > 0, "Expected warnings for small cohort"
        print(f"  Small cohort warning: {data['warnings'][0]}")
    else:
        print(f"  No warnings needed (cohort size {data['total_count']})")


def test_filter_response_time():
    """Filter should complete in < 1 second."""
    start = time.time()
    requests.post(f"{BASE_URL}/api/cohort/filter", json={"age_min": 40, "age_max": 60})
    elapsed = time.time() - start
    assert elapsed < 1.0, f"Filter took {elapsed:.2f}s, expected < 1s"
    print(f"  Response time: {elapsed*1000:.0f}ms")


if __name__ == "__main__":
    print("\n=== Test A2: Cohort Filter API Tests ===\n")
    test_filter_no_criteria()
    test_filter_reduces_cohort()
    test_filter_age_range()
    test_invalid_age_range_rejected()
    test_filter_cancer_status()
    test_filter_returns_warnings()
    test_filter_response_time()
    print("\nAll cohort filter tests passed!")
