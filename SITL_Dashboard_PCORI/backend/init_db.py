#!/usr/bin/env python3
"""
Initialize database with synthetic patient data.
Uses fixed seed for reproducibility.
"""

import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from backend.config import FEATURE_NAMES, RANDOM_SEED
from backend.database import (
    clear_patients,
    get_patient_count,
    init_schema,
    insert_patients_bulk,
)

# Set random seed
np.random.seed(RANDOM_SEED)

# Name components for generating patient names
FIRST_NAMES = [
    "James",
    "Mary",
    "Robert",
    "Patricia",
    "John",
    "Jennifer",
    "Michael",
    "Linda",
    "David",
    "Elizabeth",
    "William",
    "Barbara",
    "Richard",
    "Susan",
    "Joseph",
    "Jessica",
    "Thomas",
    "Sarah",
    "Christopher",
    "Karen",
    "Charles",
    "Lisa",
    "Daniel",
    "Nancy",
    "Matthew",
    "Betty",
    "Anthony",
    "Margaret",
    "Mark",
    "Sandra",
    "Donald",
    "Ashley",
    "Steven",
    "Kimberly",
    "Paul",
    "Emily",
    "Andrew",
    "Donna",
    "Joshua",
    "Michelle",
    "Kenneth",
    "Dorothy",
    "Kevin",
    "Carol",
    "Brian",
    "Amanda",
    "George",
    "Melissa",
]

LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Perez",
    "Thompson",
    "White",
    "Harris",
    "Sanchez",
    "Clark",
    "Ramirez",
    "Lewis",
    "Robinson",
    "Walker",
    "Young",
    "Allen",
    "King",
    "Wright",
    "Scott",
    "Torres",
    "Nguyen",
    "Hill",
    "Flores",
]


def generate_outcome(row):
    """
    Generate outcome with clear signal for testable AUROC.
    Based on clinical risk factors for OUD.
    Target event rate: 15-20%
    """
    # Logit function with clinically meaningful coefficients
    logit = (
        -2.0  # Base intercept (adjusted for ~15% baseline)
        + 0.025 * row["prior_opioid_days"]  # Higher opioid exposure = higher risk
        + 0.8 * row["benzo_use"]  # Concurrent benzo use increases risk
        + 0.15 * row["mental_health_score"]  # Mental health issues increase risk
        + 0.05 * row["er_visits"]  # ER visits slightly increase risk
        - 0.005 * row["age"]  # Slight age effect
        + 0.2 * row["cancer_status"]  # Cancer status has some effect
    )

    # Convert to probability
    prob = 1 / (1 + np.exp(-logit))

    # Generate binary outcome
    return int(np.random.random() < prob)


def generate_patients(n: int = 2000) -> list:
    """Generate n synthetic patients with realistic distributions."""

    patients = []

    for i in range(n):
        patient_id = f"P{i+1:04d}"
        name = f"{np.random.choice(FIRST_NAMES)} {np.random.choice(LAST_NAMES)}"

        # Generate features with realistic distributions
        age = int(np.clip(np.random.normal(55, 15), 18, 95))

        # Cancer status (about 15% prevalence)
        cancer_status = int(np.random.random() < 0.15)

        # Prior opioid days - bimodal distribution
        # Most patients have 0, some have moderate, few have high
        if np.random.random() < 0.4:
            prior_opioid_days = 0
        elif np.random.random() < 0.7:
            prior_opioid_days = int(np.clip(np.random.exponential(30), 1, 90))
        else:
            prior_opioid_days = int(np.clip(np.random.exponential(60) + 30, 30, 365))

        # Mental health score (0-10)
        mental_health_score = int(np.clip(np.random.exponential(2), 0, 10))

        # Benzo use (correlated with mental health and opioids)
        benzo_prob = (
            0.05 + 0.05 * mental_health_score / 10 + 0.1 * (prior_opioid_days > 30)
        )
        benzo_use = int(np.random.random() < benzo_prob)

        # ER visits (0-10, skewed towards low)
        er_visits = int(np.clip(np.random.exponential(1.5), 0, 10))

        # Create patient dict
        patient = {
            "patient_id": patient_id,
            "name": name,
            "age": age,
            "cancer_status": cancer_status,
            "prior_opioid_days": prior_opioid_days,
            "mental_health_score": mental_health_score,
            "benzo_use": benzo_use,
            "er_visits": er_visits,
            "split": "train" if np.random.random() < 0.8 else "validation",
        }

        # Generate outcome
        patient["outcome"] = generate_outcome(patient)

        patients.append(patient)

    return patients


def main():
    """Initialize database with synthetic data."""
    print("=" * 60)
    print("PCORI Dashboard - Database Initialization")
    print("=" * 60)

    # Initialize schema
    print("\n1. Creating database schema...")
    init_schema()
    print("   ✓ Schema created")

    # Check existing data
    existing_count = get_patient_count()
    if existing_count > 0:
        print(f"\n2. Found {existing_count} existing patients")
        response = input("   Clear and regenerate? (y/n): ").strip().lower()
        if response != "y":
            print("   Keeping existing data")
            return
        clear_patients()
        print("   ✓ Cleared existing data")
    else:
        print("\n2. Database is empty")

    # Generate patients
    print("\n3. Generating 2000 synthetic patients...")
    patients = generate_patients(2000)

    # Calculate statistics before inserting
    outcomes = [p["outcome"] for p in patients]
    event_rate = sum(outcomes) / len(outcomes)
    print(f"   - Event rate: {event_rate:.1%}")
    print(
        f"   - Train/Val split: {sum(1 for p in patients if p['split'] == 'train')}/{sum(1 for p in patients if p['split'] == 'validation')}"
    )

    # Insert patients
    print("\n4. Inserting patients into database...")
    insert_patients_bulk(patients)
    print("   ✓ Inserted all patients")

    # Verify
    final_count = get_patient_count()
    print(f"\n5. Verification: {final_count} patients in database")

    print("\n" + "=" * 60)
    print("Database initialization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
