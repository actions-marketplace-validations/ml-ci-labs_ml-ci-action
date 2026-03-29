"""Tests for src.validators.model_validator."""

import pytest

from src.utils.metrics import MetricsData
from src.validators.model_validator import validate_model


class TestValidateModel:
    def setup_method(self):
        self.baseline = MetricsData(
            model_name="v1",
            framework="pytorch",
            metrics={"accuracy": 0.93, "f1_score": 0.91, "loss": 0.18},
        )

    def test_improvement(self):
        current = MetricsData(
            model_name="v2",
            framework="pytorch",
            metrics={"accuracy": 0.95, "f1_score": 0.93, "loss": 0.15},
        )
        result = validate_model(current, self.baseline)
        assert not result.regression_result.regression_detected
        assert "within tolerance" in result.summary.lower() or "improved" in result.summary.lower()

    def test_regression(self):
        current = MetricsData(
            model_name="v3",
            framework="pytorch",
            metrics={"accuracy": 0.89, "f1_score": 0.85, "loss": 0.32},
        )
        result = validate_model(current, self.baseline, tolerance=0.02)
        assert result.regression_result.regression_detected
        assert "regression" in result.summary.lower()

    def test_mixed_metrics(self):
        # Accuracy improves, loss gets worse
        current = MetricsData(
            model_name="v2",
            framework="pytorch",
            metrics={"accuracy": 0.95, "f1_score": 0.93, "loss": 0.25},
        )
        result = validate_model(current, self.baseline, tolerance=0.02)
        # Loss increased from 0.18 to 0.25 (~39% increase) -> regression
        assert result.regression_result.regression_detected

    def test_identical_metrics(self):
        result = validate_model(self.baseline, self.baseline)
        assert not result.regression_result.regression_detected

    def test_new_and_removed_metrics(self):
        current = MetricsData(
            model_name="v2",
            framework="pytorch",
            metrics={"accuracy": 0.95, "new_metric": 0.99},
        )
        result = validate_model(current, self.baseline)
        assert "new_metric" in result.current_only_metrics
        assert "f1_score" in result.baseline_only_metrics
        assert "loss" in result.baseline_only_metrics

    def test_no_shared_metrics_skips_regression_detection(self):
        current = MetricsData(
            model_name="v2",
            framework="pytorch",
            metrics={"new_metric": 0.99},
        )
        result = validate_model(current, self.baseline)
        assert result.comparisons == []
        assert not result.regression_result.regression_detected
        assert "no shared metrics" in result.summary.lower()
        assert result.current_only_metrics == ["new_metric"]

    def test_model_name_and_framework(self):
        current = MetricsData(
            model_name="my-model",
            framework="sklearn",
            metrics={"accuracy": 0.95},
        )
        result = validate_model(current, self.baseline)
        assert result.model_name == "my-model"
        assert result.framework == "sklearn"

    def test_wide_tolerance(self):
        current = MetricsData(
            model_name="v3",
            framework="pytorch",
            metrics={"accuracy": 0.89, "f1_score": 0.85, "loss": 0.32},
        )
        # 50% tolerance still fails because loss regresses by ~77.8%
        result = validate_model(current, self.baseline, tolerance=0.50)
        assert result.regression_result.regression_detected

    def test_zero_tolerance(self):
        current = MetricsData(
            model_name="v2",
            framework="pytorch",
            metrics={"accuracy": 0.9299, "f1_score": 0.91, "loss": 0.18},
        )
        result = validate_model(current, self.baseline, tolerance=0.0)
        # Any decrease at all should be flagged
        assert result.regression_result.regression_detected

    def test_wilcoxon_method_routes_correctly(self):
        current = MetricsData(
            model_name="v2",
            framework="pytorch",
            metrics={"accuracy": 0.95},
            observations={"accuracy": [0.95, 0.96, 0.94, 0.95, 0.96]},
        )
        baseline = MetricsData(
            model_name="v1",
            framework="pytorch",
            metrics={"accuracy": 0.93},
            observations={"accuracy": [0.93, 0.94, 0.92, 0.93, 0.94]},
        )
        result = validate_model(current, baseline, regression_method="wilcoxon")
        assert result.regression_result.method == "wilcoxon"
        assert result.regression_result.regression_detected is False

    def test_bootstrap_method_routes_correctly(self):
        current = MetricsData(
            model_name="v2",
            framework="pytorch",
            metrics={"accuracy": 0.95},
            observations={"accuracy": [0.95, 0.96, 0.94, 0.95, 0.96]},
        )
        baseline = MetricsData(
            model_name="v1",
            framework="pytorch",
            metrics={"accuracy": 0.93},
            observations={"accuracy": [0.93, 0.94, 0.92, 0.93, 0.94]},
        )
        result = validate_model(current, baseline, regression_method="bootstrap")
        assert result.regression_result.method == "bootstrap"
        assert result.regression_result.regression_detected is False

    def test_wilcoxon_requires_observations(self):
        current = MetricsData(
            model_name="v2",
            framework="pytorch",
            metrics={"accuracy": 0.95},
        )
        with pytest.raises(ValueError, match="observations"):
            validate_model(current, self.baseline, regression_method="wilcoxon")

    def test_bootstrap_requires_observations(self):
        current = MetricsData(
            model_name="v2",
            framework="pytorch",
            metrics={"accuracy": 0.95},
        )
        with pytest.raises(ValueError, match="observations"):
            validate_model(current, self.baseline, regression_method="bootstrap")

    def test_unknown_method_raises(self):
        current = MetricsData(
            model_name="v2",
            framework="pytorch",
            metrics={"accuracy": 0.95},
        )
        with pytest.raises(ValueError, match="Unknown regression method"):
            validate_model(current, self.baseline, regression_method="invalid")
