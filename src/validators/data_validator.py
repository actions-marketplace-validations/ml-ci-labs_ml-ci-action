"""Training data quality validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass
class DataValidationResult:
    """Full result of data quality validation."""

    schema_valid: bool = True
    schema_errors: list[str] = field(default_factory=list)
    missing_value_report: dict[str, float] = field(default_factory=dict)  # column -> % missing
    duplicate_count: int = 0
    duplicate_pct: float = 0.0
    label_distribution: dict[str, int] | None = None
    drift_scores: dict[str, float] | None = None  # column -> PSI score
    drift_detected: bool = False
    overall_passed: bool = True
    details: dict[str, Any] = field(default_factory=dict)


def validate_data(
    data_path: str,
    baseline_data_path: str | None = None,
    drift_threshold: float = 0.1,
    label_column: str | None = None,
    missing_threshold: float = 0.2,
) -> DataValidationResult:
    """Run all data quality checks on training/validation data.

    Checks performed:
    1. Schema validation (if baseline provided)
    2. Missing value analysis
    3. Duplicate detection
    4. Label distribution (if label_column specified)
    5. Distribution drift via PSI (if baseline provided)

    Args:
        data_path: Path to current data file (CSV or Parquet).
        baseline_data_path: Path to baseline data for schema and drift comparison.
        drift_threshold: PSI threshold for flagging drift (default 0.1).
        label_column: Column name for label distribution analysis.
        missing_threshold: Maximum allowed missing fraction per column (default 20%).

    Returns:
        DataValidationResult with all check results.
    """
    result = DataValidationResult()

    # Load current data
    current_df = _load_dataframe(data_path)
    result.details["num_rows"] = len(current_df)
    result.details["num_columns"] = len(current_df.columns)

    # Load baseline data if provided
    baseline_df = None
    if baseline_data_path:
        try:
            baseline_df = _load_dataframe(baseline_data_path)
        except Exception as e:
            result.details["baseline_load_error"] = str(e)

    # 1. Schema validation
    if baseline_df is not None:
        result.schema_valid, result.schema_errors = _check_schema(current_df, baseline_df)

    # 2. Missing values
    result.missing_value_report = _check_missing_values(current_df)
    high_missing = [
        col for col, pct in result.missing_value_report.items() if pct > missing_threshold
    ]
    if high_missing:
        result.overall_passed = False
        result.details["high_missing_columns"] = high_missing

    # 3. Duplicates
    result.duplicate_count = int(current_df.duplicated().sum())
    result.duplicate_pct = result.duplicate_count / len(current_df) if len(current_df) > 0 else 0.0

    # 4. Label distribution
    if label_column and label_column in current_df.columns:
        result.label_distribution = current_df[label_column].value_counts().to_dict()
        # Convert numpy types to Python types for JSON serialization
        result.label_distribution = {
            str(k): int(v) for k, v in result.label_distribution.items()
        }

    # 5. Distribution drift (PSI)
    if baseline_df is not None:
        result.drift_scores = _compute_drift_scores(current_df, baseline_df)
        drifted_columns = [
            col for col, psi in result.drift_scores.items() if psi > drift_threshold
        ]
        result.drift_detected = len(drifted_columns) > 0
        if result.drift_detected:
            result.details["drifted_columns"] = drifted_columns

    # Schema failure also fails overall
    if not result.schema_valid:
        result.overall_passed = False

    return result


def _load_dataframe(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame."""
    if path.endswith(".parquet") or path.endswith(".pq"):
        return pd.read_parquet(path)
    # Default to CSV
    return pd.read_csv(path)


def _check_schema(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> tuple[bool, list[str]]:
    """Compare column names and dtypes between current and baseline data.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: list[str] = []

    current_cols = set(current_df.columns)
    baseline_cols = set(baseline_df.columns)

    missing_cols = baseline_cols - current_cols
    new_cols = current_cols - baseline_cols

    if missing_cols:
        errors.append(f"Missing columns (present in baseline): {sorted(missing_cols)}")

    if new_cols:
        # New columns are a warning, not an error — data may legitimately grow
        pass

    # Check dtype compatibility for shared columns
    shared_cols = current_cols & baseline_cols
    for col in sorted(shared_cols):
        cur_dtype = current_df[col].dtype
        base_dtype = baseline_df[col].dtype
        # Allow numeric type flexibility (int64 <-> float64)
        if _is_numeric(cur_dtype) and _is_numeric(base_dtype):
            continue
        if cur_dtype != base_dtype:
            errors.append(
                f"Column '{col}' dtype changed: {base_dtype} -> {cur_dtype}"
            )

    return len(errors) == 0, errors


def _is_numeric(dtype: np.dtype) -> bool:
    """Check if a dtype is numeric."""
    return bool(is_numeric_dtype(dtype))


def _check_missing_values(df: pd.DataFrame) -> dict[str, float]:
    """Compute missing value percentage for each column."""
    if len(df) == 0:
        return {}
    return (df.isnull().sum() / len(df)).to_dict()


def _compute_drift_scores(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute PSI (Population Stability Index) for each shared numeric column."""
    scores: dict[str, float] = {}
    shared_cols = set(current_df.columns) & set(baseline_df.columns)

    for col in sorted(shared_cols):
        if not _is_numeric(current_df[col].dtype) or not _is_numeric(baseline_df[col].dtype):
            continue

        current_series = current_df[col].dropna()
        baseline_series = baseline_df[col].dropna()

        if len(current_series) < 2 or len(baseline_series) < 2:
            continue

        scores[col] = _compute_psi(baseline_series, current_series)

    return scores


def _compute_psi(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI interpretation:
    - < 0.1: No significant shift
    - 0.1 - 0.25: Moderate shift
    - > 0.25: Significant shift

    Args:
        baseline: Baseline distribution values.
        current: Current distribution values.
        bins: Number of quantile bins (default 10).

    Returns:
        PSI score (non-negative float).
    """
    eps = 1e-4

    # Create bins from baseline quantiles
    try:
        _, bin_edges = pd.qcut(baseline, q=bins, retbins=True, duplicates="drop")
    except ValueError:
        # Not enough unique values for quantile binning — use equal-width bins
        _, bin_edges = pd.cut(baseline, bins=bins, retbins=True)

    # Compute proportions in each bin
    baseline_counts = np.histogram(baseline, bins=bin_edges)[0]
    current_counts = np.histogram(current, bins=bin_edges)[0]

    baseline_pct = baseline_counts / baseline_counts.sum()
    current_pct = current_counts / current_counts.sum()

    # Replace zeros with epsilon
    baseline_pct = np.where(baseline_pct == 0, eps, baseline_pct)
    current_pct = np.where(current_pct == 0, eps, current_pct)

    # PSI = sum((current% - baseline%) * ln(current% / baseline%))
    psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))

    return max(psi, 0.0)  # PSI should be non-negative
