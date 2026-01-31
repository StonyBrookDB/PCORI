#!/usr/bin/env python3
"""
Lab Feature Selection Script
Post-process lab abnormality summary table and produce feature-selection rankings.

Author: Auto-generated
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURABLE CONSTANTS
# =============================================================================
MIN_VALID_PATIENTS = 100
MIN_OVERALL_ABNORMAL = 0.01
MAX_OVERALL_ABNORMAL = 0.99
TOP_N = 200

# =============================================================================
# FILE PATHS
# =============================================================================
INPUT_FILE = Path("/home/zihan/pcori_project/final_data/feature_selection/labs/lab_abnormal_summary.csv")
OUTPUT_DIR = Path("/home/zihan/pcori_project/final_data/feature_selection/labs")

OUTPUT_ENRICHED = OUTPUT_DIR / "lab_abnormal_summary_enriched.csv"
OUTPUT_RANKED = OUTPUT_DIR / "lab_abnormal_ranked_patient_level.csv"
OUTPUT_TOP_N = OUTPUT_DIR / "lab_selected_top200_patient_level.csv"
LOG_FILE = OUTPUT_DIR / "lab_feature_selection.log"
FIG_LONGTAIL = OUTPUT_DIR / "lab_score_longtail.png"
FIG_COVERAGE = OUTPUT_DIR / "lab_cum_coverage.png"

# Required patient-level columns
REQUIRED_COLUMNS = [
    "n_valid_patients",
    "abnormal_rate_patients",
    "abnormal_rate_patients_pos",
    "abnormal_rate_patients_neg"
]

# Columns to include in ranked output
RANKED_OUTPUT_COLUMNS = [
    "DETAIL_LAB_PROCEDURE_ID",
    "n_valid_patients",
    "abnormal_rate_patients",
    "abnormal_rate_patients_pos",
    "abnormal_rate_patients_neg",
    "delta_abnormal_pat",
    "rr_abnormal_pat",
    "lab_score_pat",
    "n_patients_with_lab",
    "n_encounters_with_lab"
]


def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging to both file and stdout."""
    logger = logging.getLogger("lab_feature_selection")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def load_data(input_file: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load the CSV with robust dtypes."""
    logger.info(f"Loading data from: {input_file}")

    # Read with DETAIL_LAB_PROCEDURE_ID as string
    df = pd.read_csv(input_file, dtype={"DETAIL_LAB_PROCEDURE_ID": str})

    logger.info(f"Loaded {len(df)} labs from input file")

    # Convert numeric columns to float with coercion
    for col in df.columns:
        if col == "DETAIL_LAB_PROCEDURE_ID":
            continue
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif df[col].dtype in ['int64', 'int32']:
            df[col] = df[col].astype(float)

    return df


def validate_columns(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Check that required columns exist."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        error_msg = f"Missing required columns: {missing}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info("All required patient-level columns present")


def compute_derived_columns(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Compute derived columns for feature selection."""
    df = df.copy()

    # 1) delta_abnormal_pat
    df["delta_abnormal_pat"] = (
        df["abnormal_rate_patients_pos"] - df["abnormal_rate_patients_neg"]
    )

    # 2) rr_abnormal_pat (relative risk) with safe division
    df["rr_abnormal_pat"] = np.where(
        df["abnormal_rate_patients_neg"] > 0,
        df["abnormal_rate_patients_pos"] / df["abnormal_rate_patients_neg"],
        np.nan
    )

    # 3) lab_score_pat = abs(delta) * log(1 + n_valid_patients)
    df["lab_score_pat"] = (
        np.abs(df["delta_abnormal_pat"]) * np.log1p(df["n_valid_patients"])
    )

    # Log NaN counts in derived columns
    nan_counts = {
        "delta_abnormal_pat": df["delta_abnormal_pat"].isna().sum(),
        "rr_abnormal_pat": df["rr_abnormal_pat"].isna().sum(),
        "lab_score_pat": df["lab_score_pat"].isna().sum()
    }
    logger.info(f"NaN counts in derived columns: {nan_counts}")

    return df


def apply_filters(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Apply hard filters and add keep_flag column."""
    df = df.copy()

    # Create keep_flag
    df["keep_flag"] = (
        (df["n_valid_patients"] >= MIN_VALID_PATIENTS) &
        (df["abnormal_rate_patients"] >= MIN_OVERALL_ABNORMAL) &
        (df["abnormal_rate_patients"] <= MAX_OVERALL_ABNORMAL)
    )

    n_kept = df["keep_flag"].sum()
    logger.info(f"Filter criteria: n_valid_patients >= {MIN_VALID_PATIENTS}, "
                f"abnormal_rate in [{MIN_OVERALL_ABNORMAL}, {MAX_OVERALL_ABNORMAL}]")
    logger.info(f"Labs kept after filters: {n_kept} / {len(df)} "
                f"({100*n_kept/len(df):.1f}%)")

    return df


def threshold_summary(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Print threshold summary for different MIN_VALID_PATIENTS values."""
    thresholds = [10, 50, 100, 200, 500]

    logger.info("=" * 60)
    logger.info("Threshold sensitivity summary (MIN_VALID_PATIENTS):")
    logger.info("-" * 60)

    for thresh in thresholds:
        n_kept = (
            (df["n_valid_patients"] >= thresh) &
            (df["abnormal_rate_patients"] >= MIN_OVERALL_ABNORMAL) &
            (df["abnormal_rate_patients"] <= MAX_OVERALL_ABNORMAL)
        ).sum()
        logger.info(f"  MIN_VALID_PATIENTS = {thresh:4d}  =>  {n_kept:5d} labs kept")

    logger.info("=" * 60)


def log_top_labs(df_ranked: pd.DataFrame, logger: logging.Logger, n: int = 10) -> None:
    """Log the top N labs by score."""
    logger.info(f"Top {n} labs by lab_score_pat:")
    logger.info("-" * 80)

    top_n = df_ranked.head(n)
    for i, row in top_n.iterrows():
        logger.info(
            f"  {row['DETAIL_LAB_PROCEDURE_ID']:>15s} | "
            f"score={row['lab_score_pat']:.4f} | "
            f"delta={row['delta_abnormal_pat']:.4f} | "
            f"n_patients={int(row['n_valid_patients'])}"
        )
    logger.info("-" * 80)


def save_outputs(
    df_enriched: pd.DataFrame,
    df_ranked: pd.DataFrame,
    df_top_n: pd.DataFrame,
    logger: logging.Logger
) -> None:
    """Save all CSV outputs."""
    # 1) Enriched full table
    df_enriched.to_csv(OUTPUT_ENRICHED, index=False)
    logger.info(f"Saved enriched table: {OUTPUT_ENRICHED} ({len(df_enriched)} rows)")

    # 2) Filtered + ranked table (select only specified columns that exist)
    cols_to_use = [c for c in RANKED_OUTPUT_COLUMNS if c in df_ranked.columns]
    df_ranked[cols_to_use].to_csv(OUTPUT_RANKED, index=False)
    logger.info(f"Saved ranked table: {OUTPUT_RANKED} ({len(df_ranked)} rows)")

    # 3) Top-N selection table
    df_top_n[cols_to_use].to_csv(OUTPUT_TOP_N, index=False)
    logger.info(f"Saved top-{TOP_N} table: {OUTPUT_TOP_N} ({len(df_top_n)} rows)")


def generate_figures(df_ranked: pd.DataFrame, logger: logging.Logger) -> None:
    """Generate diagnostic figures."""
    if len(df_ranked) == 0:
        logger.warning("No data for figures (empty ranked table)")
        return

    # Figure 1: Lab score long-tail distribution
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    scores_sorted = df_ranked["lab_score_pat"].sort_values(ascending=False).values
    ax1.plot(range(1, len(scores_sorted) + 1), scores_sorted, 'b-', linewidth=1)
    ax1.set_yscale('log')
    ax1.set_xlabel("Rank", fontsize=12)
    ax1.set_ylabel("lab_score_pat (log scale)", fontsize=12)
    ax1.set_title("Lab Score Distribution (Long-tail)", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add vertical line at TOP_N
    if len(scores_sorted) >= TOP_N:
        ax1.axvline(x=TOP_N, color='r', linestyle='--', label=f'Top {TOP_N}')
        ax1.legend()

    fig1.savefig(FIG_LONGTAIL, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    logger.info(f"Saved figure: {FIG_LONGTAIL}")

    # Figure 2: Cumulative coverage
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    patients_sorted = df_ranked["n_valid_patients"].values
    cumsum = np.cumsum(patients_sorted)
    total_patients = cumsum[-1] if len(cumsum) > 0 else 1
    cumsum_frac = cumsum / total_patients

    ax2.plot(range(1, len(cumsum_frac) + 1), cumsum_frac, 'g-', linewidth=1.5)
    ax2.set_xlabel("Number of labs (ranked by score)", fontsize=12)
    ax2.set_ylabel("Cumulative patient coverage (fraction)", fontsize=12)
    ax2.set_title("Cumulative Patient Coverage over Ranked Labs", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # Add vertical line at TOP_N
    if len(cumsum_frac) >= TOP_N:
        coverage_at_top_n = cumsum_frac[TOP_N - 1]
        ax2.axvline(x=TOP_N, color='r', linestyle='--',
                    label=f'Top {TOP_N} ({coverage_at_top_n:.1%} coverage)')
        ax2.legend()

    fig2.savefig(FIG_COVERAGE, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"Saved figure: {FIG_COVERAGE}")


def main():
    """Main execution function."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(LOG_FILE)
    logger.info("=" * 60)
    logger.info("Lab Feature Selection Script Started")
    logger.info("=" * 60)

    try:
        # A) Load data
        df = load_data(INPUT_FILE, logger)

        # B) Validate required columns
        validate_columns(df, logger)

        # Log NaN counts in input columns
        nan_in_required = {col: df[col].isna().sum() for col in REQUIRED_COLUMNS}
        logger.info(f"NaN counts in required columns: {nan_in_required}")

        # C) Compute derived columns
        df = compute_derived_columns(df, logger)

        # D) Apply filters
        df = apply_filters(df, logger)

        # F) Threshold summary
        threshold_summary(df, logger)

        # Prepare output tables
        # Enriched table (all labs)
        df_enriched = df.copy()

        # Filtered + ranked table
        df_filtered = df[df["keep_flag"]].copy()
        df_ranked = df_filtered.sort_values("lab_score_pat", ascending=False).reset_index(drop=True)

        # Top-N table
        df_top_n = df_ranked.head(TOP_N).copy()

        # G) Log top 10 labs
        log_top_labs(df_ranked, logger, n=10)

        # E) Save outputs
        save_outputs(df_enriched, df_ranked, df_top_n, logger)

        # H) Generate figures
        generate_figures(df_ranked, logger)

        logger.info("=" * 60)
        logger.info("Lab Feature Selection Script Completed Successfully")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
