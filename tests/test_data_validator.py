"""Tests for src.validators.data_validator."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.validators.data_validator import (
    DataValidationResult,
    _compute_psi,
    _check_schema,
    validate_data,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


class TestValidateData:
    def test_basic_validation(self):
        path = os.path.join(FIXTURES_DIR, "sample_data.csv")
        result = validate_data(path)
        assert isinstance(result, DataValidationResult)
        assert result.details["num_rows"] == 30
        assert result.details["num_columns"] == 5

    def test_missing_values_detected(self):
        path = os.path.join(FIXTURES_DIR, "sample_data.csv")
        result = validate_data(path)
        # sample_data.csv has nulls in feature_a, feature_b, feature_c
        assert result.missing_value_report["feature_a"] > 0
        assert result.missing_value_report["feature_b"] > 0

    def test_duplicates_detected(self):
        path = os.path.join(FIXTURES_DIR, "sample_data.csv")
        result = validate_data(path)
        assert result.duplicate_count >= 1

    def test_with_baseline(self):
        data_path = os.path.join(FIXTURES_DIR, "sample_data.csv")
        baseline_path = os.path.join(FIXTURES_DIR, "sample_baseline_data.csv")
        result = validate_data(data_path, baseline_data_path=baseline_path)
        assert result.schema_valid  # same columns
        assert result.drift_scores is not None

    def test_label_distribution(self):
        path = os.path.join(FIXTURES_DIR, "sample_data.csv")
        result = validate_data(path, label_column="label")
        assert result.label_distribution is not None
        assert "0" in result.label_distribution or "1" in result.label_distribution

    def test_default_missing_threshold_creates_blocking_failure(self):
        path = os.path.join(FIXTURES_DIR, "sample_data.csv")
        result = validate_data(path, missing_threshold=0.02)

        assert not result.overall_passed
        assert "feature_a" in result.missing_value_failures
        assert result.failure_count >= 1

    def test_per_column_missing_threshold_override_allows_messy_column(self):
        path = os.path.join(FIXTURES_DIR, "sample_data.csv")
        result = validate_data(
            path,
            missing_threshold=0.05,
            missing_thresholds={"feature_a": 0.5, "feature_b": 0.5, "feature_c": 0.5},
        )

        assert result.overall_passed
        assert result.missing_value_failures == {}
        assert result.missing_value_thresholds["feature_a"] == 0.5

    def test_include_columns_limits_missing_value_checks(self):
        path = os.path.join(FIXTURES_DIR, "sample_data.csv")
        result = validate_data(
            path,
            include_columns=["feature_a", "label"],
            missing_threshold=0.05,
            missing_thresholds={"feature_a": 0.5},
        )

        assert result.filtered_columns == ["feature_a", "label"]
        assert "feature_b" not in result.missing_value_report
        assert result.overall_passed

    def test_exclude_columns_suppresses_irrelevant_columns(self):
        path = os.path.join(FIXTURES_DIR, "sample_data.csv")
        result = validate_data(
            path,
            exclude_columns=["feature_a", "feature_b", "feature_c"],
            missing_threshold=0.0,
        )

        assert result.filtered_columns == ["category", "label"]
        assert result.overall_passed

    def test_label_shift_warns_when_distribution_moves(self):
        current_df = pd.DataFrame(
            {
                "feature_a": list(range(10)),
                "label": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            }
        )
        baseline_df = pd.DataFrame(
            {
                "feature_a": list(range(10)),
                "label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as current_file:
            current_df.to_csv(current_file.name, index=False)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as baseline_file:
            baseline_df.to_csv(baseline_file.name, index=False)

        try:
            result = validate_data(
                current_file.name,
                baseline_data_path=baseline_file.name,
                label_column="label",
            )
        finally:
            os.unlink(current_file.name)
            os.unlink(baseline_file.name)

        assert result.label_distribution is not None
        assert result.baseline_label_distribution is not None
        assert result.label_shift_detected
        assert any("Label distribution shifted" in warning for warning in result.warnings)

    def test_zero_config_guidance_mentions_label_candidates_and_baseline(self):
        path = os.path.join(FIXTURES_DIR, "sample_data.csv")
        result = validate_data(path)
        guidance = result.details["guidance"]

        assert any("Detected candidate label column" in note for note in guidance)
        assert any("No `baseline-data-path` provided" in note for note in guidance)

    def test_schema_mismatch(self):
        """Test with mismatched schemas."""
        # Create a temp file with different columns
        df = pd.DataFrame({"col_x": [1, 2, 3], "col_y": [4, 5, 6]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            baseline_path = os.path.join(FIXTURES_DIR, "sample_baseline_data.csv")
            result = validate_data(f.name, baseline_data_path=baseline_path)
            assert not result.schema_valid
            assert len(result.schema_errors) > 0
        os.unlink(f.name)


class TestCheckSchema:
    def test_identical_schemas(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7.0, 8.0]})
        valid, errors, warnings = _check_schema(df1, df2)
        assert valid
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_missing_column(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        valid, errors, warnings = _check_schema(df1, df2)
        assert not valid
        assert any("missing baseline column" in e.lower() for e in errors)
        assert warnings == []

    def test_dtype_change_string_to_int(self):
        df1 = pd.DataFrame({"a": ["x", "y"]})
        df2 = pd.DataFrame({"a": [1, 2]})
        valid, errors, warnings = _check_schema(df1, df2)
        assert not valid
        assert any("type changed" in e for e in errors)
        assert warnings == []

    def test_numeric_type_flexibility(self):
        """int64 <-> float64 should be allowed."""
        df1 = pd.DataFrame({"a": [1, 2]})  # int64
        df2 = pd.DataFrame({"a": [1.0, 2.0]})  # float64
        valid, errors, warnings = _check_schema(df1, df2)
        assert valid
        assert warnings == []

    def test_new_columns_are_actionable_warning(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2]})
        valid, errors, warnings = _check_schema(df1, df2)

        assert valid
        assert errors == []
        assert any("new column" in warning.lower() for warning in warnings)


class TestComputePSI:
    def test_identical_distributions(self):
        data = pd.Series(np.random.normal(0, 1, 1000), name="test")
        psi = _compute_psi(data, data)
        assert psi < 0.05  # Should be very low

    def test_shifted_distribution(self):
        baseline = pd.Series(np.random.normal(0, 1, 1000), name="test")
        current = pd.Series(np.random.normal(3, 1, 1000), name="test")  # big shift
        psi = _compute_psi(baseline, current)
        assert psi > 0.25  # Should indicate significant drift

    def test_non_negative(self):
        baseline = pd.Series(np.random.normal(0, 1, 500), name="test")
        current = pd.Series(np.random.normal(0.1, 1.1, 500), name="test")
        psi = _compute_psi(baseline, current)
        assert psi >= 0.0

    def test_small_sample(self):
        baseline = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="test")
        current = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5], name="test")
        # Should not crash with small samples
        psi = _compute_psi(baseline, current, bins=3)
        assert isinstance(psi, float)
        assert psi >= 0.0

    def test_constant_values(self):
        baseline = pd.Series([1.0] * 100, name="test")
        current = pd.Series([1.0] * 100, name="test")
        # Should handle constant (zero-variance) data
        psi = _compute_psi(baseline, current, bins=5)
        assert isinstance(psi, float)
