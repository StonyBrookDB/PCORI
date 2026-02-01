"""
Generate synthetic patient data for SITL prototype.
200 patients, 5 features, OD (opioid overdose) outcome.
"""

import json
import os
import random

# Seed for reproducibility
random.seed(42)

FIRST_NAMES = [
    "James",
    "Mary",
    "John",
    "Patricia",
    "Robert",
    "Jennifer",
    "Michael",
    "Linda",
    "William",
    "Elizabeth",
    "David",
    "Barbara",
    "Richard",
    "Susan",
    "Joseph",
    "Jessica",
    "Thomas",
    "Sarah",
    "Charles",
    "Karen",
    "Daniel",
    "Nancy",
    "Matthew",
    "Lisa",
    "Anthony",
    "Betty",
    "Mark",
    "Margaret",
    "Donald",
    "Sandra",
    "Steven",
    "Ashley",
    "Paul",
    "Kimberly",
    "Andrew",
    "Emily",
    "Joshua",
    "Donna",
    "Kenneth",
    "Michelle",
    "Kevin",
    "Dorothy",
    "Brian",
    "Carol",
    "George",
    "Amanda",
    "Edward",
    "Melissa",
    "Ronald",
    "Deborah",
    "Timothy",
    "Stephanie",
    "Jason",
    "Rebecca",
    "Jeffrey",
    "Sharon",
    "Ryan",
    "Laura",
    "Jacob",
    "Cynthia",
    "Gary",
    "Kathleen",
    "Nicholas",
    "Amy",
    "Eric",
    "Angela",
    "Jonathan",
    "Shirley",
    "Stephen",
    "Anna",
    "Larry",
    "Brenda",
    "Justin",
    "Pamela",
    "Scott",
    "Emma",
    "Brandon",
    "Nicole",
    "Benjamin",
    "Helen",
    "Samuel",
    "Samantha",
    "Raymond",
    "Katherine",
    "Gregory",
    "Christine",
    "Frank",
    "Debra",
    "Alexander",
    "Rachel",
    "Patrick",
    "Carolyn",
    "Jack",
    "Janet",
    "Dennis",
    "Catherine",
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
    "Green",
    "Adams",
    "Nelson",
    "Baker",
    "Hall",
    "Rivera",
    "Campbell",
    "Mitchell",
    "Carter",
    "Roberts",
]


def generate_patients(n=200):
    """
    Generate n synthetic patients with:
    - 5 features relevant to OD risk
    - Binary outcome (OD within 1 year)
    """
    patients = []

    for i in range(n):
        # Generate features that logically relate to OD risk
        age = random.randint(18, 75)

        # Prior opioid prescriptions (days in past year) - major risk factor
        prior_opioid_days = random.choices(
            [0, random.randint(1, 30), random.randint(31, 90), random.randint(91, 365)],
            weights=[0.4, 0.3, 0.2, 0.1],
        )[0]

        # Mental health score (0-10, higher = more issues) - risk factor
        mental_health_score = random.choices(
            range(11),
            weights=[0.15, 0.15, 0.15, 0.12, 0.1, 0.1, 0.08, 0.06, 0.04, 0.03, 0.02],
        )[0]

        # Concurrent benzodiazepine use (0 or 1) - major risk factor
        benzo_use = random.choices([0, 1], weights=[0.8, 0.2])[0]

        # ER visits in past year - correlates with risk
        er_visits = random.choices(
            [0, 1, 2, 3, 4, 5], weights=[0.4, 0.25, 0.15, 0.1, 0.06, 0.04]
        )[0]

        # Calculate OD probability based on features (realistic-ish model)
        # Base rate ~5%, modified by risk factors
        log_odds = -3.0  # Base (roughly 5% baseline)
        log_odds += 0.02 * prior_opioid_days  # More opioid days = higher risk
        log_odds += 0.3 * mental_health_score  # Mental health issues increase risk
        log_odds += 1.5 * benzo_use  # Benzo + opioid is dangerous
        log_odds += 0.4 * er_visits  # ER visits correlate
        log_odds += -0.01 * age + 0.5  # Younger slightly higher risk

        # Add noise
        log_odds += random.gauss(0, 0.5)

        # Convert to probability
        import math

        prob = 1 / (1 + math.exp(-log_odds))

        # Generate outcome
        od_outcome = 1 if random.random() < prob else 0

        patient = {
            "id": f"P{i+1:03d}",
            "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "features": {
                "age": age,
                "prior_opioid_days": prior_opioid_days,
                "mental_health_score": mental_health_score,
                "benzo_use": benzo_use,
                "er_visits": er_visits,
            },
            "outcome": od_outcome,  # 1 = OD within 1 year, 0 = no OD
            "outcome_probability": round(prob, 4),
        }
        patients.append(patient)

    return patients


def split_train_val(patients, n_train=150):
    """Split into train (150) and validation (50)."""
    random.shuffle(patients)
    train = patients[:n_train]
    val = patients[n_train:]

    # Mark split
    for p in train:
        p["split"] = "train"
    for p in val:
        p["split"] = "validation"

    return train, val


def main():
    patients = generate_patients(200)
    train, val = split_train_val(patients)

    all_patients = train + val
    # Sort by ID for consistent ordering
    all_patients.sort(key=lambda x: x["id"])

    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "patients.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "patients": all_patients,
                "feature_descriptions": {
                    "age": "Patient age in years",
                    "prior_opioid_days": "Number of days with opioid prescriptions in past year",
                    "mental_health_score": "Mental health severity score (0-10, higher = more severe)",
                    "benzo_use": "Concurrent benzodiazepine use (0=No, 1=Yes)",
                    "er_visits": "Number of ER visits in past year",
                },
                "outcome_description": "Opioid overdose within 1 year (0=No, 1=Yes)",
                "splits": {"train": len(train), "validation": len(val)},
            },
            f,
            indent=2,
        )

    # Print summary
    train_od = sum(1 for p in train if p["outcome"] == 1)
    val_od = sum(1 for p in val if p["outcome"] == 1)
    print(f"Generated {len(all_patients)} patients")
    print(
        f"Train: {len(train)} patients, {train_od} OD cases ({100*train_od/len(train):.1f}%)"
    )
    print(
        f"Validation: {len(val)} patients, {val_od} OD cases ({100*val_od/len(val):.1f}%)"
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
