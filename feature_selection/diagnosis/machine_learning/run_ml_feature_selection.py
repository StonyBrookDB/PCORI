#!/usr/bin/env python3
"""
================================================================================
Machine Learning Feature Selection Pipeline for ICD-10 Diagnosis Codes
================================================================================

This script performs feature selection/ranking using two models:
1. Elastic Net Logistic Regression (sklearn)
2. LightGBM with SHAP importance (or XGBoost as fallback)

Input: Long-format CSV with diagnosis codes per encounter
Output: Ranked feature lists from both models

Key Design Decisions:
- Feature encoding: Binary presence (1 if diagnosis appears in encounter, 0 otherwise)
- Sample unit: Encounter (one row per ENCOUNTER_ID in feature matrix)
- Label semantics: LABEL is patient-level (all encounters of a patient share the same label)
- Train/Val split: Patient-level (GroupShuffleSplit) to prevent data leakage
- Sparse matrices: Used throughout for memory efficiency

================================================================================
"""

import argparse
import os
import sys
import logging
import warnings
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str) -> str:
    """Setup logging to both console and file."""
    log_file = os.path.join(output_dir, "ml_feature_selection.log")

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    return log_file


def log(msg: str, level: str = "info"):
    """Log a message."""
    if level == "info":
        logging.info(msg)
    elif level == "warning":
        logging.warning(msg)
    elif level == "error":
        logging.error(msg)


# =============================================================================
# Data Loading and Validation
# =============================================================================

def load_and_validate_data(input_csv: str) -> pd.DataFrame:
    """
    Load CSV and validate required columns and data integrity.

    Note: LABEL is patient-level (not encounter-level). Each patient has a single
    label that applies to all their encounters. This function validates label
    consistency at the patient level and propagates labels to all rows.
    """
    log(f"Loading data from: {input_csv}")

    required_cols = ["PATIENT_ID", "ENCOUNTER_ID", "ICD10_3Digits", "LABEL"]

    # Load with string types for IDs
    df = pd.read_csv(
        input_csv,
        dtype={
            "PATIENT_ID": str,
            "ENCOUNTER_ID": str,
            "ICD10_3Digits": str,
            "LABEL": int
        },
        usecols=required_cols
    )

    log(f"  Raw data shape: {df.shape}")

    # Check required columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Drop rows with missing values
    n_before = len(df)
    df = df.dropna(subset=required_cols)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        log(f"  Dropped {n_dropped:,} rows with missing values", "warning")

    # Validate LABEL is binary
    unique_labels = df["LABEL"].unique()
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"LABEL must be binary (0/1). Found: {unique_labels}")

    # LABEL is patient-level: validate consistency at PATIENT_ID level
    log("  LABEL is patient-level; validating consistency within patients...")
    label_per_patient = df.groupby("PATIENT_ID")["LABEL"].nunique()
    inconsistent = label_per_patient[label_per_patient > 1]
    if len(inconsistent) > 0:
        raise ValueError(
            f"Found {len(inconsistent)} patients with inconsistent labels. "
            f"Examples: {inconsistent.head().index.tolist()}"
        )

    # Construct patient-level label map and propagate to all rows
    log("  Propagating patient-level labels to all encounters...")
    patient_label = df.groupby("PATIENT_ID")["LABEL"].first().astype(int)
    df = df.drop(columns=["LABEL"])
    df = df.merge(patient_label.rename("LABEL"), on="PATIENT_ID", how="left")
    df["LABEL"] = df["LABEL"].astype(int)

    log(f"  Unique patients:   {df['PATIENT_ID'].nunique():,}")
    log(f"  Unique encounters: {df['ENCOUNTER_ID'].nunique():,}")
    log(f"  Unique ICD10 codes:{df['ICD10_3Digits'].nunique():,}")
    log(f"  Patient-level label distribution: 0={patient_label.value_counts().get(0, 0):,}, 1={patient_label.value_counts().get(1, 0):,}")

    return df


# =============================================================================
# Feature Matrix Construction
# =============================================================================

def build_sparse_feature_matrix(
    df: pd.DataFrame
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Build encounter-level sparse feature matrix with binary presence encoding.

    Returns:
        X: sparse CSR matrix (n_encounters, n_features)
        y: label array (n_encounters,)
        encounter_ids: array of encounter IDs
        patient_ids: array of patient IDs (for grouping)
        feature_names: dict mapping feature index to ICD10 code
    """
    log("Building sparse feature matrix...")

    # Deduplicate (ENCOUNTER_ID, ICD10_3Digits) - binary presence
    df_dedup = df.drop_duplicates(subset=["ENCOUNTER_ID", "ICD10_3Digits"])
    log(f"  After dedup: {len(df_dedup):,} (encounter, diagnosis) pairs")

    # Create mappings
    unique_encounters = df_dedup["ENCOUNTER_ID"].unique()
    unique_features = df_dedup["ICD10_3Digits"].unique()

    encounter_to_idx = {enc: i for i, enc in enumerate(unique_encounters)}
    feature_to_idx = {feat: i for i, feat in enumerate(unique_features)}
    idx_to_feature = {i: feat for feat, i in feature_to_idx.items()}

    n_encounters = len(unique_encounters)
    n_features = len(unique_features)

    log(f"  Matrix dimensions: {n_encounters:,} encounters x {n_features:,} features")

    # Build sparse matrix using COO format then convert to CSR
    rows = df_dedup["ENCOUNTER_ID"].map(encounter_to_idx).values
    cols = df_dedup["ICD10_3Digits"].map(feature_to_idx).values
    data = np.ones(len(rows), dtype=np.float32)

    X = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(n_encounters, n_features),
        dtype=np.float32
    ).tocsr()

    # Get labels and patient IDs per encounter (take first occurrence)
    encounter_info = df.drop_duplicates(subset=["ENCOUNTER_ID"])[
        ["ENCOUNTER_ID", "LABEL", "PATIENT_ID"]
    ].set_index("ENCOUNTER_ID")

    # Align with matrix row order
    y = encounter_info.loc[unique_encounters, "LABEL"].values.astype(np.int32)
    patient_ids = encounter_info.loc[unique_encounters, "PATIENT_ID"].values

    log(f"  X shape: {X.shape}, nnz: {X.nnz:,}, density: {X.nnz / (X.shape[0] * X.shape[1]):.6f}")
    log(f"  y shape: {y.shape}, class balance: {y.mean():.4f}")

    return X, y, unique_encounters, patient_ids, idx_to_feature


# =============================================================================
# Train/Validation Split
# =============================================================================

def patient_level_split(
    X: sparse.csr_matrix,
    y: np.ndarray,
    patient_ids: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data at patient level to prevent leakage.

    Returns:
        X_train, X_val, y_train, y_val, train_idx, val_idx
    """
    log(f"Performing patient-level split (test_size={test_size})...")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=patient_ids))

    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    train_patients = len(np.unique(patient_ids[train_idx]))
    val_patients = len(np.unique(patient_ids[val_idx]))

    log(f"  Train: {len(train_idx):,} encounters from {train_patients:,} patients")
    log(f"  Val:   {len(val_idx):,} encounters from {val_patients:,} patients")
    log(f"  Train label balance: {y_train.mean():.4f}")
    log(f"  Val label balance:   {y_val.mean():.4f}")

    return X_train, X_val, y_train, y_val, train_idx, val_idx


# =============================================================================
# Model A: Elastic Net Logistic Regression
# =============================================================================

def train_elastic_net(
    X_train: sparse.csr_matrix,
    X_val: sparse.csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    idx_to_feature: Dict[int, str],
    output_dir: str,
    top_n: int,
    seed: int
) -> pd.DataFrame:
    """
    Train Elastic Net Logistic Regression with hyperparameter search.

    Returns DataFrame of top features ranked by absolute coefficient.
    """
    log("\n" + "=" * 70)
    log("MODEL A: Elastic Net Logistic Regression")
    log("=" * 70)

    # Scale features (MaxAbsScaler is sparse-compatible, no centering)
    log("Scaling features with MaxAbsScaler...")
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Hyperparameter grid
    l1_ratios = [0.2, 0.5, 0.8]
    C_values = [0.01, 0.1, 1.0, 10.0]

    log("Hyperparameter search...")
    log(f"  l1_ratio grid: {l1_ratios}")
    log(f"  C grid: {C_values}")

    best_auroc = -1
    best_params = None
    best_model = None

    for l1_ratio in l1_ratios:
        for C in C_values:
            model = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=l1_ratio,
                C=C,
                max_iter=1000,
                random_state=seed,
                n_jobs=-1,
                warm_start=False
            )

            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            auroc = roc_auc_score(y_val, y_pred_proba)

            log(f"    l1_ratio={l1_ratio}, C={C}: Val AUROC = {auroc:.4f}")

            if auroc > best_auroc:
                best_auroc = auroc
                best_params = {"l1_ratio": l1_ratio, "C": C}
                best_model = model

    log(f"\nBest params: {best_params}")
    log(f"Best Val AUROC: {best_auroc:.4f}")

    # Extract coefficients
    coefs = best_model.coef_.flatten()

    # Build results DataFrame
    results = []
    for idx, coef in enumerate(coefs):
        feature = idx_to_feature[idx]
        results.append({
            "feature": feature,
            "coef": coef,
            "abs_coef": abs(coef)
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    results_df["rank"] = range(1, len(results_df) + 1)
    results_df = results_df[["rank", "feature", "coef", "abs_coef"]]

    # Save full results
    full_path = os.path.join(output_dir, "elastic_net_all_features.csv")
    results_df.to_csv(full_path, index=False)
    log(f"Saved all features to: {full_path}")

    # Save top N
    top_df = results_df.head(top_n).copy()
    top_path = os.path.join(output_dir, "top_features_elastic_net.csv")
    top_df.to_csv(top_path, index=False)
    log(f"Saved top {top_n} features to: {top_path}")

    # Print top 20
    log("\nTop 20 features by |coefficient|:")
    log(top_df.head(20).to_string(index=False))

    # Count non-zero coefficients
    n_nonzero = np.sum(coefs != 0)
    log(f"\nNon-zero coefficients: {n_nonzero} / {len(coefs)}")

    return top_df


# =============================================================================
# Model B: LightGBM with SHAP
# =============================================================================

def train_lightgbm_with_shap(
    X_train: sparse.csr_matrix,
    X_val: sparse.csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    idx_to_feature: Dict[int, str],
    output_dir: str,
    top_n: int,
    seed: int,
    shap_sample_size: int = 5000
) -> pd.DataFrame:
    """
    Train LightGBM and compute SHAP importance.
    Falls back to XGBoost if LightGBM not available.

    Returns DataFrame of top features ranked by mean absolute SHAP value.
    """
    log("\n" + "=" * 70)
    log("MODEL B: LightGBM with SHAP Importance")
    log("=" * 70)

    # Try to import LightGBM, fall back to XGBoost
    try:
        import lightgbm as lgb
        use_lightgbm = True
        log("Using LightGBM")
    except ImportError:
        log("LightGBM not available, falling back to XGBoost", "warning")
        try:
            import xgboost as xgb
            use_lightgbm = False
            log("Using XGBoost")
        except ImportError:
            raise ImportError("Neither LightGBM nor XGBoost is available. Please install one.")

    # Import SHAP
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is not available. Please install: pip install shap")

    feature_names = [idx_to_feature[i] for i in range(len(idx_to_feature))]

    if use_lightgbm:
        # LightGBM training
        log("Training LightGBM...")

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "seed": seed,
            "verbose": -1,
            "n_jobs": -1
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        # Validation AUROC
        y_pred_proba = model.predict(X_val)
        auroc = roc_auc_score(y_val, y_pred_proba)
        log(f"Val AUROC: {auroc:.4f}")
        log(f"Best iteration: {model.best_iteration}")

    else:
        # XGBoost training
        log("Training XGBoost...")

        import xgboost as xgb

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": seed,
            "verbosity": 0
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=100
        )

        # Validation AUROC
        y_pred_proba = model.predict(dval)
        auroc = roc_auc_score(y_val, y_pred_proba)
        log(f"Val AUROC: {auroc:.4f}")
        log(f"Best iteration: {model.best_iteration}")

    # SHAP importance
    log(f"\nComputing SHAP values (sample size: {shap_sample_size})...")

    # Sample validation set for SHAP computation
    n_val = X_val.shape[0]
    if n_val > shap_sample_size:
        np.random.seed(seed)
        shap_idx = np.random.choice(n_val, shap_sample_size, replace=False)
        X_shap = X_val[shap_idx]
    else:
        X_shap = X_val

    n_samples = X_shap.shape[0]
    n_features = X_shap.shape[1]  # Use X_shap shape as ground truth
    log(f"  X_shap shape: {X_shap.shape}")
    log(f"  n_samples: {n_samples}, n_features: {n_features}")

    # Create SHAP explainer and compute values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap, check_additivity=False)

    log(f"  Raw shap_values type: {type(shap_values)}")
    if hasattr(shap_values, 'shape'):
        log(f"  Raw shap_values shape: {shap_values.shape}")

    # =========================================================================
    # Normalize shap_values to dense 2-D numpy array: (n_samples, n_features)
    # =========================================================================

    # Case 1: List of arrays (binary classification returns [class0, class1])
    if isinstance(shap_values, list):
        log(f"  SHAP returned list of length {len(shap_values)}")
        for i, arr in enumerate(shap_values):
            arr_shape = arr.shape if hasattr(arr, 'shape') else 'N/A'
            arr_type = type(arr).__name__
            log(f"    shap_values[{i}] type: {arr_type}, shape: {arr_shape}")
        # For binary classification, take class 1 (positive class)
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    # Case 2: SHAP Explanation object (newer SHAP versions)
    if hasattr(shap_values, 'values'):
        log("  SHAP returned Explanation object, extracting .values")
        shap_values = shap_values.values

    # Case 3: Handle sparse matrices (LightGBM often returns sparse)
    if sparse.issparse(shap_values):
        log(f"  SHAP returned sparse matrix: {type(shap_values).__name__}, shape: {shap_values.shape}")
        shap_values_dense = shap_values.toarray()
        log(f"  Converted to dense: {shap_values_dense.shape}")
    else:
        shap_values_dense = np.asarray(shap_values, dtype=np.float64)
        log(f"  shap_values_dense shape: {shap_values_dense.shape}, dtype: {shap_values_dense.dtype}")

    # Case 4: Handle 3-D arrays (e.g., shape (n_samples, n_features, n_classes))
    if shap_values_dense.ndim == 3:
        log(f"  Detected 3D array with shape {shap_values_dense.shape}, selecting class dimension")
        # If shape is (n_samples, n_features, 2), take class 1
        if shap_values_dense.shape[2] == 2:
            shap_values_dense = shap_values_dense[:, :, 1]
        elif shap_values_dense.shape[2] == 1:
            shap_values_dense = shap_values_dense[:, :, 0]
        else:
            shap_values_dense = np.squeeze(shap_values_dense)
        log(f"  After 3D handling: {shap_values_dense.shape}")

    # Case 5: Handle 1-D arrays (shouldn't happen, but be safe)
    if shap_values_dense.ndim == 1:
        log(f"  Detected 1D array, reshaping to (1, n_features)")
        shap_values_dense = shap_values_dense.reshape(1, -1)

    # Case 6: Extra bias/base-value column (shape n_features + 1)
    if shap_values_dense.ndim == 2 and shap_values_dense.shape[1] == n_features + 1:
        log(f"  Detected extra bias column, trimming from {shap_values_dense.shape[1]} to {n_features}")
        shap_values_dense = shap_values_dense[:, :n_features]

    log(f"  Final shap_values_dense shape: {shap_values_dense.shape}")

    # Validate final shape
    if shap_values_dense.ndim != 2:
        raise ValueError(
            f"SHAP normalization failed: expected 2D, got ndim={shap_values_dense.ndim}, shape={shap_values_dense.shape}"
        )
    if shap_values_dense.shape[1] != n_features:
        raise ValueError(
            f"SHAP shape mismatch: got {shap_values_dense.shape[1]} columns, expected {n_features} features. "
            f"Full shape: {shap_values_dense.shape}"
        )

    # =========================================================================
    # Compute mean absolute SHAP value per feature
    # =========================================================================
    mean_abs_shap = np.mean(np.abs(shap_values_dense), axis=0)

    # Force to 1-D array and ensure float64
    mean_abs_shap = np.asarray(mean_abs_shap, dtype=np.float64).flatten()

    log(f"  Final mean_abs_shap shape: {mean_abs_shap.shape}, expected: ({n_features},)")

    # Final validation
    if mean_abs_shap.shape != (n_features,):
        raise ValueError(
            f"mean_abs_shap shape mismatch: got {mean_abs_shap.shape}, expected ({n_features},)"
        )

    # =========================================================================
    # Build results DataFrame
    # =========================================================================
    results = [
        {"feature": feature_names[i], "mean_abs_shap": float(mean_abs_shap[i])}
        for i in range(n_features)
    ]

    results_df = pd.DataFrame(results)
    log(f"  results_df has {len(results_df)} rows (expected {n_features})")

    if len(results_df) != n_features:
        raise ValueError(
            f"Results DataFrame has {len(results_df)} rows, expected {n_features}"
        )

    results_df = results_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    results_df["rank"] = range(1, len(results_df) + 1)
    results_df = results_df[["rank", "feature", "mean_abs_shap"]]

    # Save full results
    model_name = "lightgbm" if use_lightgbm else "xgboost"
    full_path = os.path.join(output_dir, f"{model_name}_all_features_shap.csv")
    results_df.to_csv(full_path, index=False)
    log(f"Saved all features to: {full_path} ({len(results_df)} rows)")

    # Save top N
    top_df = results_df.head(top_n).copy()
    top_path = os.path.join(output_dir, f"top_features_{model_name}.csv")
    top_df.to_csv(top_path, index=False)
    log(f"Saved top {min(top_n, len(results_df))} features to: {top_path}")

    # Print top 20
    log(f"\nTop 20 features by mean |SHAP|:")
    log(top_df.head(20).to_string(index=False))

    return top_df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ML-based feature selection for ICD-10 diagnosis codes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/home/zihan/pcori_project/final_data/feature_selection/diagnosis.csv",
        help="Path to input CSV (long format with PATIENT_ID, ENCOUNTER_ID, ICD10_3Digits, LABEL)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/zihan/pcori_project/final_data/feature_selection/machine_learning",
        help="Output directory for results"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=1000,
        help="Number of top features to save"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--shap_sample_size",
        type=int,
        default=5000,
        help="Number of samples for SHAP computation"
    )
    parser.add_argument(
        "--skip_elastic_net",
        action="store_true",
        help="Skip Elastic Net model"
    )
    parser.add_argument(
        "--skip_lightgbm",
        action="store_true",
        help="Skip LightGBM/XGBoost model"
    )

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    log_file = setup_logging(args.output_dir)

    log("=" * 70)
    log("ML FEATURE SELECTION PIPELINE")
    log("=" * 70)
    log(f"Input CSV:    {args.input_csv}")
    log(f"Output dir:   {args.output_dir}")
    log(f"Log file:     {log_file}")
    log(f"Top N:        {args.top_n}")
    log(f"Seed:         {args.seed}")
    log(f"SHAP samples: {args.shap_sample_size}")
    log("=" * 70)

    start_time = datetime.now()

    try:
        # Load and validate data
        df = load_and_validate_data(args.input_csv)

        # Build sparse feature matrix
        X, y, encounter_ids, patient_ids, idx_to_feature = build_sparse_feature_matrix(df)

        # Free memory
        del df

        # Train/validation split (patient-level)
        X_train, X_val, y_train, y_val, train_idx, val_idx = patient_level_split(
            X, y, patient_ids, test_size=0.2, seed=args.seed
        )

        # Model A: Elastic Net
        if not args.skip_elastic_net:
            elastic_net_results = train_elastic_net(
                X_train, X_val, y_train, y_val,
                idx_to_feature, args.output_dir, args.top_n, args.seed
            )
        else:
            log("\nSkipping Elastic Net model (--skip_elastic_net)")

        # Model B: LightGBM/XGBoost with SHAP
        if not args.skip_lightgbm:
            lgbm_results = train_lightgbm_with_shap(
                X_train, X_val, y_train, y_val,
                idx_to_feature, args.output_dir, args.top_n, args.seed,
                args.shap_sample_size
            )
        else:
            log("\nSkipping LightGBM/XGBoost model (--skip_lightgbm)")

        elapsed = datetime.now() - start_time
        log("\n" + "=" * 70)
        log(f"PIPELINE COMPLETE")
        log(f"Total time: {elapsed}")
        log("=" * 70)

    except Exception as e:
        log(f"ERROR: {str(e)}", "error")
        import traceback
        log(traceback.format_exc(), "error")
        sys.exit(1)


if __name__ == "__main__":
    main()
