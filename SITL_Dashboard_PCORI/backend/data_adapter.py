"""
Data Adapter: Unified interface for loading different datasets.

Provides consistent API for:
- synthetic (PCORI OUD prediction)
- breast_cancer (sklearn)
- diabetes (sklearn, regression)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    dataset_id: str
    display_name: str
    description: str
    task_type: str  # "classification" or "regression"
    features: List[str]
    feature_descriptions: Dict[str, str]
    target_name: str
    target_description: str
    sample_count: int


# Feature descriptions for each dataset
SYNTHETIC_FEATURE_DESCRIPTIONS = {
    "age": "Patient age in years",
    "prior_opioid_days": "Days of prior opioid prescriptions in past year",
    "mental_health_score": "Mental health assessment score (0-10)",
    "benzo_use": "Benzodiazepine co-prescription (0 or 1)",
    "er_visits": "Emergency room visits in past year",
}

BREAST_CANCER_FEATURE_DESCRIPTIONS = {
    "mean radius": "Mean of distances from center to points on the perimeter",
    "mean texture": "Standard deviation of gray-scale values",
    "mean perimeter": "Mean size of the core tumor perimeter",
    "mean area": "Mean area of the core tumor",
    "mean smoothness": "Mean of local variation in radius lengths",
    "mean compactness": "Mean of perimeter^2 / area - 1.0",
    "mean concavity": "Mean of severity of concave portions of the contour",
    "mean concave points": "Mean number of concave portions of the contour",
    "mean symmetry": "Mean symmetry of the tumor",
    "mean fractal dimension": "Mean coastline approximation - 1",
    "radius error": "Standard error of radius",
    "texture error": "Standard error of texture",
    "perimeter error": "Standard error of perimeter",
    "area error": "Standard error of area",
    "smoothness error": "Standard error of smoothness",
    "compactness error": "Standard error of compactness",
    "concavity error": "Standard error of concavity",
    "concave points error": "Standard error of concave points",
    "symmetry error": "Standard error of symmetry",
    "fractal dimension error": "Standard error of fractal dimension",
    "worst radius": "Worst (largest) radius",
    "worst texture": "Worst (largest) texture",
    "worst perimeter": "Worst (largest) perimeter",
    "worst area": "Worst (largest) area",
    "worst smoothness": "Worst (largest) smoothness",
    "worst compactness": "Worst (largest) compactness",
    "worst concavity": "Worst (largest) concavity",
    "worst concave points": "Worst (largest) concave points",
    "worst symmetry": "Worst (largest) symmetry",
    "worst fractal dimension": "Worst (largest) fractal dimension",
}

DIABETES_FEATURE_DESCRIPTIONS = {
    "age": "Age in years (normalized)",
    "sex": "Sex (normalized)",
    "bmi": "Body mass index (normalized)",
    "bp": "Average blood pressure (normalized)",
    "s1": "T-Cells (normalized)",
    "s2": "Low-density lipoproteins (normalized)",
    "s3": "High-density lipoproteins (normalized)",
    "s4": "Thyroid stimulating hormone (normalized)",
    "s5": "Lamotrigine (normalized)",
    "s6": "Blood sugar level (normalized)",
}


def get_dataset_config(dataset_id: str) -> DatasetConfig:
    """Get configuration for a dataset."""
    if dataset_id == "synthetic":
        return DatasetConfig(
            dataset_id="synthetic",
            display_name="Synthetic Patients (OUD Risk)",
            description="2000 synthetic patient records for opioid use disorder and overdose risk prediction",
            task_type="classification",
            features=list(SYNTHETIC_FEATURE_DESCRIPTIONS.keys()),
            feature_descriptions=SYNTHETIC_FEATURE_DESCRIPTIONS,
            target_name="outcome",
            target_description="Opioid overdose occurred (0=No, 1=Yes)",
            sample_count=2000,
        )
    elif dataset_id == "breast_cancer":
        return DatasetConfig(
            dataset_id="breast_cancer",
            display_name="Breast Cancer Diagnosis",
            description="569 samples of breast cancer cell nuclei measurements for malignancy prediction",
            task_type="classification",
            features=list(BREAST_CANCER_FEATURE_DESCRIPTIONS.keys()),
            feature_descriptions=BREAST_CANCER_FEATURE_DESCRIPTIONS,
            target_name="target",
            target_description="Diagnosis (0=Malignant, 1=Benign)",
            sample_count=569,
        )
    elif dataset_id == "diabetes":
        return DatasetConfig(
            dataset_id="diabetes",
            display_name="Diabetes Progression",
            description="442 diabetes patients with disease progression measure after one year",
            task_type="regression",
            features=list(DIABETES_FEATURE_DESCRIPTIONS.keys()),
            feature_descriptions=DIABETES_FEATURE_DESCRIPTIONS,
            target_name="target",
            target_description="Disease progression (quantitative measure)",
            sample_count=442,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_id}. Available: synthetic, breast_cancer, diabetes"
        )


def load_dataset(dataset_id: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load a dataset and return (X, y, feature_names).

    Returns:
        X: Feature DataFrame
        y: Target Series
        feature_names: List of feature column names
    """
    if dataset_id == "synthetic":
        return _load_synthetic()
    elif dataset_id == "breast_cancer":
        return _load_breast_cancer()
    elif dataset_id == "diabetes":
        return _load_diabetes()
    else:
        raise ValueError(f"Unknown dataset: {dataset_id}")


def _load_synthetic() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load synthetic PCORI dataset."""
    try:
        raise ImportError("Using fallback for report-v2")  # from yl.data.pcori.synthetic import FEATURE_NAMES, generate_patients_dataframe

        df = generate_patients_dataframe(n=2000, seed=42, version="v2")
        X = df[list(FEATURE_NAMES)]
        y = df["outcome"]
        return X, y, list(FEATURE_NAMES)
    except ImportError:
        # Fallback: generate simple synthetic data
        np.random.seed(42)
        n = 2000
        features = list(SYNTHETIC_FEATURE_DESCRIPTIONS.keys())
        X = pd.DataFrame(
            {
                "age": np.random.randint(18, 85, n),
                "prior_opioid_days": np.random.randint(0, 365, n),
                "mental_health_score": np.random.randint(0, 11, n),
                "benzo_use": np.random.randint(0, 2, n),
                "er_visits": np.random.randint(0, 10, n),
            }
        )
        # Simple outcome logic
        risk = (
            X["prior_opioid_days"] / 365 * 0.3
            + X["mental_health_score"] / 10 * 0.2
            + X["benzo_use"] * 0.2
            + X["er_visits"] / 10 * 0.3
        )
        y = (risk > np.random.random(n) * 0.5).astype(int)
        return X, pd.Series(y, name="outcome"), features


def _load_breast_cancer() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load breast cancer dataset from sklearn."""
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    return X, y, list(X.columns)


def _load_diabetes() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load diabetes dataset from sklearn."""
    from sklearn.datasets import load_diabetes

    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target
    return X, y, list(X.columns)


def get_samples_for_display(
    dataset_id: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get samples formatted for dashboard display.

    Returns list of dicts with:
        - id: sample identifier
        - features: dict of feature values
        - target: target value
        - split: "train" or "validation"
    """
    X, y, feature_names = load_dataset(dataset_id)

    # Create train/val split (80/20)
    np.random.seed(42)
    n = len(X)
    indices = np.random.permutation(n)
    train_size = int(0.8 * n)
    train_indices = set(indices[:train_size])

    samples = []
    for i in range(n):
        if limit and len(samples) >= limit:
            break

        sample = {
            "id": f"{dataset_id}_{i}",
            "index": i,
            "features": {col: float(X.iloc[i][col]) for col in feature_names},
            "target": float(y.iloc[i]),
            "split": "train" if i in train_indices else "validation",
        }
        samples.append(sample)

    return samples


def get_training_data(
    dataset_id: str,
    features: List[str],
    sample_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get training data for model fitting.

    Args:
        dataset_id: Dataset to load
        features: List of feature names to include
        sample_indices: Optional subset of sample indices

    Returns:
        X_train, y_train, X_val, y_val as numpy arrays
    """
    X, y, _ = load_dataset(dataset_id)

    # Filter to requested features
    X = X[features]

    # Filter to requested samples
    if sample_indices is not None:
        X = X.iloc[sample_indices]
        y = y.iloc[sample_indices]

    # Create train/val split
    np.random.seed(42)
    n = len(X)
    indices = np.random.permutation(n)
    train_size = int(0.8 * n)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    X_train = X.iloc[train_idx].values.astype(np.float32)
    y_train = y.iloc[train_idx].values.astype(np.float32)
    X_val = X.iloc[val_idx].values.astype(np.float32)
    y_val = y.iloc[val_idx].values.astype(np.float32)

    return X_train, y_train, X_val, y_val


# Available datasets (medical/healthcare only)
AVAILABLE_DATASETS = ["synthetic", "breast_cancer", "diabetes"]

# Note: Other sklearn datasets (iris, wine, digits, california_housing) are available
# in yl.data.datasets but not exposed in the dashboard for focus on medical use cases.
