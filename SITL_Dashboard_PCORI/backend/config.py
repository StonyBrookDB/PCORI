"""
Configuration for SITL Dashboard (report-v2 - standalone version)

This version removes yl.* package dependencies by hardcoding constants
and using direct file paths for data loading.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Database configuration
DB_PATH = os.environ.get("PCORI_DB_PATH", str(BASE_DIR / "data" / "patients.db"))

# API configuration
API_PORT = int(os.environ.get("PCORI_API_PORT", "8501"))
API_HOST = os.environ.get("PCORI_API_HOST", "0.0.0.0")

# Random seed for reproducibility
RANDOM_SEED = 42

# Training constraints
MIN_COHORT_SIZE = 100
MIN_EVENT_COUNT = 10

# SHAP configuration
SHAP_TIMEOUT_SECONDS = 30
SHAP_MAX_SAMPLES = 500  # Max samples for global SHAP computation

# Application version
VERSION = "0.2.0-report"

# Feature definitions (for synthetic/demo data)
FEATURE_NAMES = [
    "age",
    "prior_opioid_days",
    "mental_health_score",
    "benzo_use",
    "er_visits",
]

FEATURE_DESCRIPTIONS = {
    "age": "Patient age in years",
    "prior_opioid_days": "Number of days with opioid prescriptions in past year",
    "mental_health_score": "Mental health severity score (0-10)",
    "benzo_use": "Concurrent benzodiazepine use (0=No, 1=Yes)",
    "er_visits": "Number of ER visits in past year",
}


# ==================== Health Facts Sequential Constants ====================
# These were previously imported from yl.data.pcori.health_facts_seq_committers

# Supported sequence lengths
SUPPORTED_T = [5, 10, 20]

# Supported normalizations
SUPPORTED_NORMALIZATIONS = ["standard", "minmax", "none"]

# Supported sample fractions for caching
SUPPORTED_SAMPLES = [1.0, 0.1, 0.01]

# Default label column
DEFAULT_LABEL = "oud_od"

# Code version for cache key
CODE_VERSION = "v1"

# Base path for Health Facts middleware cache
HEALTH_FACTS_MIDDLEWARE_PATH = Path(
    os.environ.get(
        "HEALTH_FACTS_MIDDLEWARE_PATH",
        "/path/to/health_facts_middleware"
    )
)


# ==================== Feature Sets ====================
# These were previously from yl.data.pcori.health_facts_feature_sets


@dataclass(frozen=True)
class FeatureSet:
    """
    Immutable definition of a predefined feature set.

    Attributes:
        name: Unique identifier for this feature set (used in committer keys)
        feature_ids: List of feature_id values to include
        description: Human-readable description
    """
    name: str
    feature_ids: List[int]
    description: str

    def __len__(self) -> int:
        return len(self.feature_ids)


def _load_features_table(features_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load features.csv from middleware directory.

    Args:
        features_path: Optional override path to features.csv

    Returns:
        DataFrame with columns: feature_id, name, group, description
    """
    path = features_path or (HEALTH_FACTS_MIDDLEWARE_PATH / "features.csv")
    if not path.exists():
        raise FileNotFoundError(f"features.csv not found at {path}")

    df = pd.read_csv(path)

    # Validate required columns
    if "feature_id" not in df.columns:
        raise ValueError("features.csv must include a 'feature_id' column")
    if "group" not in df.columns:
        # Fallback if grouping is absent
        df["group"] = "unknown"

    # Drop rows with NA feature_id and ensure integer type
    df = df.dropna(subset=["feature_id"]).copy()
    df["feature_id"] = df["feature_id"].astype(int)

    return df


def _select_by_group(features_df: pd.DataFrame, group_name: str) -> List[int]:
    """Select all feature_ids belonging to a specific group."""
    return features_df.loc[
        features_df["group"] == group_name, "feature_id"
    ].astype(int).tolist()


def _top_k_by_group(
    features_df: pd.DataFrame,
    group_name: str,
    k: int
) -> List[int]:
    """
    Select top-k feature_ids from a group.

    Uses deterministic ordering by feature_id (first k features).
    """
    group_df = features_df.loc[features_df["group"] == group_name]
    return group_df["feature_id"].astype(int).head(k).tolist()


def get_feature_sets(features_path: Optional[Path] = None) -> Dict[str, FeatureSet]:
    """
    Get all predefined feature sets.

    Args:
        features_path: Optional override path to features.csv

    Returns:
        Dictionary mapping feature set name to FeatureSet object
    """
    try:
        df = _load_features_table(features_path)
    except FileNotFoundError:
        # Return empty dict if features.csv not available
        return {}

    result: Dict[str, FeatureSet] = {}

    # Build demographics feature set
    demo_fids = _select_by_group(df, "demographic")
    if demo_fids:
        result["demographics"] = FeatureSet(
            name="demographics",
            feature_ids=demo_fids,
            description=f"Demographic features ({len(demo_fids)} features)"
        )

    # Build temporal feature set
    temp_fids = _select_by_group(df, "temporal")
    if temp_fids:
        result["temporal"] = FeatureSet(
            name="temporal",
            feature_ids=temp_fids,
            description=f"Temporal features ({len(temp_fids)} features)"
        )

    # Build top 100 diagnoses feature set
    diag100_fids = _top_k_by_group(df, "diagnosis", k=100)
    if diag100_fids:
        result["diagnoses_top100"] = FeatureSet(
            name="diagnoses_top100",
            feature_ids=diag100_fids,
            description=f"Top 100 diagnoses ({len(diag100_fids)} features)"
        )

    # Build all diagnoses feature set
    diag_all_fids = _select_by_group(df, "diagnosis")
    if diag_all_fids:
        result["diagnoses_all"] = FeatureSet(
            name="diagnoses_all",
            feature_ids=diag_all_fids,
            description=f"All diagnoses ({len(diag_all_fids)} features)"
        )

    # Build procedures feature set
    proc_fids = _select_by_group(df, "procedure")
    if proc_fids:
        result["procedures"] = FeatureSet(
            name="procedures",
            feature_ids=proc_fids,
            description=f"Procedure features ({len(proc_fids)} features)"
        )

    # Build core feature set: demographics + temporal + top 100 diagnoses
    core_fids = sorted(set(demo_fids + temp_fids + diag100_fids))
    if core_fids:
        result["core"] = FeatureSet(
            name="core",
            feature_ids=core_fids,
            description=f"Core features: demographics + temporal + top 100 diagnoses ({len(core_fids)} features)"
        )

    # Build full feature set
    full_fids = df["feature_id"].astype(int).tolist()
    if full_fids:
        result["full"] = FeatureSet(
            name="full",
            feature_ids=full_fids,
            description=f"All features ({len(full_fids)} features)"
        )

    return result


def list_feature_set_names(features_path: Optional[Path] = None) -> List[str]:
    """
    List all available feature set names.

    Returns:
        List of feature set names
    """
    return list(get_feature_sets(features_path).keys())
