"""
Health Facts Sequential Data Loader (report-v2 - standalone version)

Replaces yl.data.pcori.health_facts_sequential_api with direct numpy loading
from pre-cached files.

This module provides the same API as the original but without yl.* dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from .config import (
    SUPPORTED_T,
    SUPPORTED_NORMALIZATIONS,
    SUPPORTED_SAMPLES,
    DEFAULT_LABEL,
    CODE_VERSION,
    HEALTH_FACTS_MIDDLEWARE_PATH,
    get_feature_sets,
    list_feature_set_names,
)


@dataclass
class SequencesArtifact:
    """Metadata about cached sequence arrays."""
    root_dir: str
    feature_set: str
    feature_ids: List[int]
    T: int
    D: int
    normalization: str
    label_name: str
    sample_fraction: float
    code_version: str
    X_train_path: str
    y_train_path: str
    X_val_path: str
    y_val_path: str
    X_test_path: str
    y_test_path: str
    n_train: int
    n_val: int
    n_test: int


def get_health_facts_sequences(
    T: int = 10,
    feature_set: str = "core",
    normalization: Literal["standard", "minmax", "none"] = "standard",
    sample_fraction: float = 1.0,
    label: str = DEFAULT_LABEL,
    code_version: str = CODE_VERSION,
    load_arrays: bool = True,
) -> Dict[str, Any]:
    """
    Load pre-cached Health Facts sequential data.

    This is a simplified version that loads directly from disk cache
    without the committer system.

    Args:
        T: Sequence length (number of encounters per patient)
        feature_set: Name of predefined feature set to use
        normalization: Normalization mode ("standard", "minmax", "none")
        sample_fraction: Fraction of data to use (1.0, 0.1, or 0.01)
        label: Label column name
        code_version: Code version for cache key
        load_arrays: If True, load numpy arrays into memory. If False, only
            return paths to arrays (for memory-mapped loading)

    Returns:
        Dictionary containing:
        - X_train, y_train: Training arrays (if load_arrays=True)
        - X_val, y_val: Validation arrays
        - X_test, y_test: Test arrays
        - feature_set: Name of feature set
        - feature_ids: List of feature IDs included
        - T: Sequence length
        - D: Feature dimension
        - normalization: Normalization mode used
        - label: Label column name
        - sample_fraction: Sample fraction used
        - n_train, n_val, n_test: Sample counts
        - artifact_root: Path to cached artifacts
        - X_train_path, y_train_path, etc.: Paths to numpy arrays

    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If cache not available
    """
    # Validate parameters
    if T not in SUPPORTED_T:
        raise ValueError(f"T={T} not supported. Use one of: {SUPPORTED_T}")
    if normalization not in SUPPORTED_NORMALIZATIONS:
        raise ValueError(
            f"normalization='{normalization}' not supported. "
            f"Use one of: {SUPPORTED_NORMALIZATIONS}"
        )
    if sample_fraction not in SUPPORTED_SAMPLES:
        raise ValueError(
            f"sample_fraction={sample_fraction} not supported. "
            f"Use one of: {SUPPORTED_SAMPLES}"
        )

    available_sets = list_feature_set_names()
    if feature_set not in available_sets:
        # If features.csv not available, allow core/full as defaults
        if feature_set not in ["core", "full", "demographics", "temporal", 
                               "diagnoses_top100", "diagnoses_all", "procedures"]:
            raise ValueError(
                f"feature_set='{feature_set}' not found. "
                f"Available: {available_sets or ['core', 'full']}"
            )

    # Build cache path
    edges_dir = HEALTH_FACTS_MIDDLEWARE_PATH / f"_committed_edges_col_{code_version}_sf{sample_fraction}"
    seq_dir = edges_dir / f"seq_T{T}_{feature_set}_{normalization}"

    # Check if cache exists
    X_train_path = seq_dir / "X_train.npy"
    if not X_train_path.exists():
        raise FileNotFoundError(
            f"Sequences not cached for T={T}, feature_set={feature_set}, "
            f"normalization={normalization}, sample_fraction={sample_fraction}. "
            f"Expected path: {seq_dir}"
        )

    # Load feature set info
    fs = get_feature_sets().get(feature_set)
    feature_ids = fs.feature_ids if fs else []

    # Get dimensions from mmap'd array
    X_train_mmap = np.load(X_train_path, mmap_mode="r")
    n_train = X_train_mmap.shape[0]
    D = X_train_mmap.shape[-1] if len(X_train_mmap.shape) > 2 else 0

    # Get val/test counts
    y_val_path = seq_dir / "y_val.npy"
    y_test_path = seq_dir / "y_test.npy"
    n_val = int(np.load(y_val_path, mmap_mode="r").shape[0]) if y_val_path.exists() else 0
    n_test = int(np.load(y_test_path, mmap_mode="r").shape[0]) if y_test_path.exists() else 0

    artifact = SequencesArtifact(
        root_dir=str(seq_dir),
        feature_set=feature_set,
        feature_ids=feature_ids,
        T=T,
        D=D,
        normalization=normalization,
        label_name=label,
        sample_fraction=sample_fraction,
        code_version=code_version,
        X_train_path=str(seq_dir / "X_train.npy"),
        y_train_path=str(seq_dir / "y_train.npy"),
        X_val_path=str(seq_dir / "X_val.npy"),
        y_val_path=str(seq_dir / "y_val.npy"),
        X_test_path=str(seq_dir / "X_test.npy"),
        y_test_path=str(seq_dir / "y_test.npy"),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )

    result = {
        # Metadata
        "feature_set": artifact.feature_set,
        "feature_ids": artifact.feature_ids,
        "T": artifact.T,
        "D": artifact.D,
        "normalization": artifact.normalization,
        "label": artifact.label_name,
        "sample_fraction": artifact.sample_fraction,
        "code_version": artifact.code_version,
        "n_train": artifact.n_train,
        "n_val": artifact.n_val,
        "n_test": artifact.n_test,
        "artifact_root": artifact.root_dir,
        # Paths
        "X_train_path": artifact.X_train_path,
        "y_train_path": artifact.y_train_path,
        "X_val_path": artifact.X_val_path,
        "y_val_path": artifact.y_val_path,
        "X_test_path": artifact.X_test_path,
        "y_test_path": artifact.y_test_path,
    }

    if load_arrays:
        result["X_train"] = np.load(artifact.X_train_path)
        result["y_train"] = np.load(artifact.y_train_path)
        result["X_val"] = np.load(artifact.X_val_path)
        result["y_val"] = np.load(artifact.y_val_path)
        result["X_test"] = np.load(artifact.X_test_path)
        result["y_test"] = np.load(artifact.y_test_path)

    return result


def get_health_facts_sequences_mmap(
    T: int = 10,
    feature_set: str = "core",
    normalization: Literal["standard", "minmax", "none"] = "standard",
    sample_fraction: float = 1.0,
    label: str = DEFAULT_LABEL,
    code_version: str = CODE_VERSION,
) -> Dict[str, Any]:
    """
    Load Health Facts sequences as memory-mapped arrays.

    This is more memory-efficient for very large datasets as it doesn't
    load the entire array into memory at once.

    Same signature as get_health_facts_sequences, but returns memory-mapped
    numpy arrays instead of fully-loaded arrays.
    """
    data = get_health_facts_sequences(
        T=T,
        feature_set=feature_set,
        normalization=normalization,
        sample_fraction=sample_fraction,
        label=label,
        code_version=code_version,
        load_arrays=False,
    )

    # Load as memory-mapped
    data["X_train"] = np.load(data["X_train_path"], mmap_mode="r")
    data["y_train"] = np.load(data["y_train_path"], mmap_mode="r")
    data["X_val"] = np.load(data["X_val_path"], mmap_mode="r")
    data["y_val"] = np.load(data["y_val_path"], mmap_mode="r")
    data["X_test"] = np.load(data["X_test_path"], mmap_mode="r")
    data["y_test"] = np.load(data["y_test_path"], mmap_mode="r")

    return data


def list_available_feature_sets() -> List[str]:
    """
    List all available feature set names.

    Returns:
        List of feature set names that can be used with get_health_facts_sequences
    """
    return list_feature_set_names()


def get_cache_status(
    T: int = 10,
    feature_set: str = "core",
    normalization: str = "standard",
    sample_fraction: float = 1.0,
    label: str = DEFAULT_LABEL,
    code_version: str = CODE_VERSION,
) -> Dict[str, Any]:
    """
    Check if data is cached for given parameters.

    Returns:
        Dictionary with 'cached' boolean and path info
    """
    edges_dir = HEALTH_FACTS_MIDDLEWARE_PATH / f"_committed_edges_col_{code_version}_sf{sample_fraction}"
    seq_dir = edges_dir / f"seq_T{T}_{feature_set}_{normalization}"
    X_train_path = seq_dir / "X_train.npy"

    return {
        "cached": X_train_path.exists(),
        "path": str(seq_dir),
        "T": T,
        "feature_set": feature_set,
        "normalization": normalization,
        "sample_fraction": sample_fraction,
    }
