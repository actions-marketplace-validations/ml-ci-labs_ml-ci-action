"""Training data quality validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

DEFAULT_MISSING_THRESHOLD = 0.2
LABEL_IMBALANCE_SHIFT_THRESHOLD = 0.10
COMMON_LABEL_COLUMNS = ("label", "labels", "target", "class", "y")


@dataclass
class DataValidationResult:
    """Full result of data quality validation."""

    schema_valid: bool = True
    schema_errors: list[str] = field(default_factory=list)
    schema_warnings: list[str] = field(default_factory=list)
    missing_value_report: dict[str, float] = field(default_factory=dict)  # column -> % missing
    missing_value_failures: dict[str, float] = field(default_factory=dict)
    missing_value_thresholds: dict[str, float] = field(default_factory=dict)
    duplicate_count: int = 0
    duplicate_pct: float = 0.0
    label_column: str | None = None
    label_distribution: dict[str, int] | None = None
    baseline_label_distribution: dict[str, int] | None = None
    label_distribution_shift: dict[str, float] | None = None
    label_shift_detected: bool = False
    drift_scores: dict[str, float] | None = None  # column -> PSI score
    drift_detected: bool = False
    filtered_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    overall_passed: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def warning_count(self) -> int:
        """Return the number of non-blocking warnings."""
        return len(self.warnings)

    @property
    def failure_count(self) -> int:
        """Return the number of blocking failures."""
        return len(self.failures)


def validate_data(
    data_path: str,
    baseline_data_path: str | None = None,
    drift_threshold: float = 0.1,
    label_column: str | None = None,
    missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
    missing_thresholds: dict[str, float] | None = None,
    include_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
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
        missing_thresholds: Optional per-column missing-value thresholds.
        include_columns: Optional allowlist of columns to validate.
        exclude_columns: Optional blocklist of columns to ignore.

    Returns:
        DataValidationResult with all check results.
    """
    result = DataValidationResult()
    guidance: list[str] = []
    missing_thresholds = missing_thresholds or {}
    include_columns = include_columns or []
    exclude_columns = exclude_columns or []

    # Load current data
    current_df = _load_dataframe(data_path)
    result.details["num_rows"] = len(current_df)
    result.details["num_columns"] = len(current_df.columns)
    result.details["original_columns"] = list(current_df.columns)

    current_df, current_filter_details = _filter_columns(
        current_df,
        include_columns=include_columns,
        exclude_columns=exclude_columns,
    )
    result.filtered_columns = list(current_df.columns)
    result.details["filtered_num_columns"] = len(current_df.columns)
    result.details["applied_include_columns"] = current_filter_details["applied_include_columns"]
    result.details["applied_exclude_columns"] = current_filter_details["applied_exclude_columns"]
    if current_filter_details["missing_include_columns"]:
        result.warnings.append(
            "Configured include columns were not found in current data: "
            + ", ".join(f"`{col}`" for col in current_filter_details["missing_include_columns"])
        )

    if include_columns or exclude_columns:
        result.details["column_scope"] = (
            f"Checked {len(current_df.columns)} column(s) after applying include/exclude filters."
        )
    else:
        guidance.append(
            "Checking all columns by default. Use `policy.data.include_columns` or "
            "`policy.data.exclude_columns` to focus validation on relevant features."
        )

    label_candidates = _detect_label_candidates(current_df.columns)
    if label_column is None and label_candidates:
        guidance.append(
            "Detected candidate label column(s): "
            + ", ".join(f"`{col}`" for col in label_candidates)
            + ". Set `policy.data.label_column` to track class balance."
        )

    # Load baseline data if provided
    baseline_df = None
    if baseline_data_path:
        try:
            baseline_df = _load_dataframe(baseline_data_path)
            baseline_df, baseline_filter_details = _filter_columns(
                baseline_df,
                include_columns=include_columns,
                exclude_columns=exclude_columns,
            )
            if baseline_filter_details["missing_include_columns"]:
                result.warnings.append(
                    "Configured include columns were not found in baseline data: "
                    + ", ".join(f"`{col}`" for col in baseline_filter_details["missing_include_columns"])
                )
        except Exception as e:
            result.details["baseline_load_error"] = str(e)
            result.warnings.append(
                f"Could not load baseline data from `{baseline_data_path}`; schema and drift checks were skipped."
            )
    else:
        guidance.append(
            "No `baseline-data-path` provided, so schema comparison, drift checks, and label shift detection were skipped."
        )

    # 1. Schema validation
    if baseline_df is not None:
        result.schema_valid, result.schema_errors, result.schema_warnings = _check_schema(
            current_df,
            baseline_df,
        )
        result.warnings.extend(result.schema_warnings)
        if result.schema_errors:
            result.failures.extend(result.schema_errors)

    # 2. Missing values
    result.missing_value_report = _check_missing_values(current_df)
    result.missing_value_thresholds = {
        col: float(missing_thresholds.get(col, missing_threshold)) for col in current_df.columns
    }
    result.missing_value_failures = {
        col: pct
        for col, pct in result.missing_value_report.items()
        if pct > result.missing_value_thresholds.get(col, missing_threshold)
    }
    if result.missing_value_failures:
        result.failures.append(
            "Missing-value thresholds exceeded for "
            + ", ".join(
                f"`{col}` ({pct:.1%} > {result.missing_value_thresholds[col]:.1%})"
                for col, pct in sorted(result.missing_value_failures.items(), key=lambda item: -item[1])
            )
        )
        result.details["high_missing_columns"] = list(result.missing_value_failures)
    else:
        guidance.append(
            f"Missing-value checks used the default threshold of {missing_threshold:.0%}"
            " for all validated columns."
            if not missing_thresholds
            else "Missing-value checks used the configured default threshold plus per-column overrides."
        )

    # 3. Duplicates
    result.duplicate_count = int(current_df.duplicated().sum())
    result.duplicate_pct = result.duplicate_count / len(current_df) if len(current_df) > 0 else 0.0
    if result.duplicate_count > 0:
        result.warnings.append(
            f"Found {result.duplicate_count} duplicate row(s) ({result.duplicate_pct:.1%} of filtered data)."
        )

    # 4. Label distribution
    result.label_column = label_column
    if label_column:
        if label_column in current_df.columns:
            result.label_distribution = _compute_label_distribution(current_df[label_column])
            if baseline_df is not None:
                if label_column in baseline_df.columns:
                    result.baseline_label_distribution = _compute_label_distribution(
                        baseline_df[label_column]
                    )
                    result.label_distribution_shift = _compute_label_distribution_shift(
                        result.label_distribution,
                        result.baseline_label_distribution,
                    )
                    shifted_labels = {
                        label: delta
                        for label, delta in result.label_distribution_shift.items()
                        if abs(delta) > LABEL_IMBALANCE_SHIFT_THRESHOLD
                    }
                    result.label_shift_detected = bool(shifted_labels)
                    if shifted_labels:
                        result.warnings.append(
                            "Label distribution shifted by more than "
                            f"{LABEL_IMBALANCE_SHIFT_THRESHOLD:.0%} for "
                            + ", ".join(
                                f"`{label}` ({delta:+.1%})"
                                for label, delta in sorted(
                                    shifted_labels.items(),
                                    key=lambda item: -abs(item[1]),
                                )
                            )
                        )
                        result.details["label_shifted_classes"] = shifted_labels
                else:
                    result.warnings.append(
                        f"Configured label column `{label_column}` is missing from baseline data; label shift check was skipped."
                    )
        else:
            result.warnings.append(
                f"Configured label column `{label_column}` is missing from current data; label checks were skipped."
            )

    # 5. Distribution drift (PSI)
    if baseline_df is not None:
        result.drift_scores = _compute_drift_scores(current_df, baseline_df)
        drifted_columns = [
            col for col, psi in result.drift_scores.items() if psi > drift_threshold
        ]
        result.drift_detected = len(drifted_columns) > 0
        if result.drift_detected:
            result.details["drifted_columns"] = drifted_columns
            result.warnings.append(
                "Distribution drift exceeded the PSI threshold for "
                + ", ".join(
                    f"`{col}` ({result.drift_scores[col]:.3f})"
                    for col in sorted(drifted_columns, key=lambda name: -result.drift_scores[name])
                )
            )

    result.details["guidance"] = guidance

    # Blocking failures determine pass/fail.
    if result.failures:
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
) -> tuple[bool, list[str], list[str]]:
    """Compare column names and dtypes between current and baseline data.

    Returns:
        (is_valid, blocking_errors, warnings)
    """
    errors: list[str] = []
    warnings: list[str] = []

    current_cols = set(current_df.columns)
    baseline_cols = set(baseline_df.columns)

    missing_cols = baseline_cols - current_cols
    new_cols = current_cols - baseline_cols

    if missing_cols:
        errors.append(
            "Current data is missing baseline column(s): "
            + ", ".join(f"`{col}`" for col in sorted(missing_cols))
            + ". Add them back, exclude them with `policy.data.exclude_columns`, or refresh the baseline if the schema change is intentional."
        )

    if new_cols:
        warnings.append(
            "Current data has new column(s) not present in the baseline: "
            + ", ".join(f"`{col}`" for col in sorted(new_cols))
            + ". Add them to the baseline or exclude them if they are intentionally noisy."
        )

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
                f"Column `{col}` type changed from `{base_dtype}` to `{cur_dtype}`. "
                "Cast it back before exporting the dataset, exclude the column, or refresh the baseline if the change is intentional."
            )

    return len(errors) == 0, errors, warnings


def _is_numeric(dtype: np.dtype) -> bool:
    """Check if a dtype is numeric."""
    return bool(is_numeric_dtype(dtype))


def _check_missing_values(df: pd.DataFrame) -> dict[str, float]:
    """Compute missing value percentage for each column."""
    if len(df) == 0:
        return {}
    return (df.isnull().sum() / len(df)).to_dict()


def _filter_columns(
    df: pd.DataFrame,
    include_columns: list[str],
    exclude_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Apply include/exclude filters while preserving column order."""
    filtered_columns = list(df.columns)
    missing_include_columns: list[str] = []

    if include_columns:
        include_set = set(include_columns)
        filtered_columns = [col for col in filtered_columns if col in include_set]
        missing_include_columns = sorted(include_set - set(df.columns))

    if exclude_columns:
        exclude_set = set(exclude_columns)
        filtered_columns = [col for col in filtered_columns if col not in exclude_set]

    return (
        df.loc[:, filtered_columns],
        {
            "applied_include_columns": [col for col in include_columns if col in df.columns],
            "applied_exclude_columns": [col for col in exclude_columns if col in df.columns],
            "missing_include_columns": missing_include_columns,
        },
    )


def _compute_label_distribution(series: pd.Series) -> dict[str, int]:
    """Return a JSON-safe label distribution."""
    counts = series.value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _compute_label_distribution_shift(
    current_counts: dict[str, int],
    baseline_counts: dict[str, int],
) -> dict[str, float]:
    """Return current-minus-baseline share deltas for each label."""
    current_total = sum(current_counts.values())
    baseline_total = sum(baseline_counts.values())
    all_labels = sorted(set(current_counts) | set(baseline_counts))

    shifts: dict[str, float] = {}
    for label in all_labels:
        current_share = current_counts.get(label, 0) / current_total if current_total else 0.0
        baseline_share = baseline_counts.get(label, 0) / baseline_total if baseline_total else 0.0
        shifts[label] = current_share - baseline_share
    return shifts


def _detect_label_candidates(columns: pd.Index) -> list[str]:
    """Return likely label columns for zero-config guidance."""
    normalized = {str(column).lower(): str(column) for column in columns}
    return [normalized[name] for name in COMMON_LABEL_COLUMNS if name in normalized]


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
