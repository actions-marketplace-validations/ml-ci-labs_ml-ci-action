"""Tests for src.utils.metrics."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.utils.metrics import (
    BaselineFetchError,
    MetricsData,
    _is_higher_better,
    compare_metrics,
    load_metrics,
    load_metrics_from_github,
    validate_paired_observations,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


class TestLoadMetrics:
    def test_load_valid_file(self):
        path = os.path.join(FIXTURES_DIR, "current_metrics.json")
        data = load_metrics(path)
        assert isinstance(data, MetricsData)
        assert data.model_name == "test-classifier-v2"
        assert data.framework == "pytorch"
        assert "accuracy" in data.metrics
        assert data.metrics["accuracy"] == 0.95

    def test_load_baseline_file(self):
        path = os.path.join(FIXTURES_DIR, "baseline_metrics.json")
        data = load_metrics(path)
        assert data.model_name == "test-classifier-v1"
        assert data.metrics["accuracy"] == 0.93

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_metrics("/nonexistent/path/metrics.json")

    def test_load_missing_metrics_key(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model_name": "test"}, f)
            f.flush()
            with pytest.raises(ValueError, match="metrics"):
                load_metrics(f.name)
        os.unlink(f.name)

    def test_load_non_numeric_metric(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"metrics": {"accuracy": "high"}}, f)
            f.flush()
            with pytest.raises(ValueError, match="numeric"):
                load_metrics(f.name)
        os.unlink(f.name)

    def test_load_minimal_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"metrics": {"accuracy": 0.9}}, f)
            f.flush()
            data = load_metrics(f.name)
            assert data.model_name == "unnamed-model"
            assert data.framework == "unknown"
            assert data.metrics["accuracy"] == 0.9
        os.unlink(f.name)


class TestLoadMetricsFromGithub:
    @patch("src.utils.metrics.requests.get")
    def test_missing_file_raises_file_not_found(self, mock_get):
        response = MagicMock()
        response.status_code = 404
        mock_get.return_value = response

        with pytest.raises(FileNotFoundError, match="not found"):
            load_metrics_from_github(
                repo="owner/repo",
                path="metrics.json",
                ref="main",
                token="token123",
            )

    @patch("src.utils.metrics.requests.get")
    def test_permission_or_size_failure_raises_actionable_error(self, mock_get):
        response = MagicMock()
        response.status_code = 403
        response.json.return_value = {"message": "This API returns blobs up to 1 MB"}
        mock_get.return_value = response

        with pytest.raises(BaselineFetchError, match="Use a local file path"):
            load_metrics_from_github(
                repo="owner/repo",
                path="metrics.json",
                ref="main",
                token="token123",
            )


class TestIsHigherBetter:
    def test_accuracy_higher_is_better(self):
        assert _is_higher_better("accuracy") is True

    def test_loss_lower_is_better(self):
        assert _is_higher_better("loss") is False

    def test_mse_lower_is_better(self):
        assert _is_higher_better("mse") is False

    def test_f1_higher_is_better(self):
        assert _is_higher_better("f1_score") is True

    def test_unknown_defaults_higher(self):
        assert _is_higher_better("custom_metric") is True

    def test_unknown_with_loss_keyword(self):
        assert _is_higher_better("custom_loss") is False

    def test_unknown_with_error_keyword(self):
        assert _is_higher_better("prediction_error") is False


class TestCompareMetrics:
    def setup_method(self):
        self.current = MetricsData(
            model_name="v2",
            metrics={"accuracy": 0.95, "f1_score": 0.93, "loss": 0.15},
        )
        self.baseline = MetricsData(
            model_name="v1",
            metrics={"accuracy": 0.93, "f1_score": 0.91, "loss": 0.18},
        )

    def test_improvement_detected(self):
        comparisons = compare_metrics(self.current, self.baseline)
        acc = next(c for c in comparisons if c.name == "accuracy")
        assert acc.improved is True
        assert acc.regression is False
        assert acc.delta > 0

    def test_loss_improvement(self):
        comparisons = compare_metrics(self.current, self.baseline)
        loss = next(c for c in comparisons if c.name == "loss")
        assert loss.improved is True  # loss decreased = improvement
        assert loss.higher_is_better is False
        assert loss.delta < 0

    def test_regression_detected(self):
        regressed = MetricsData(
            model_name="bad",
            metrics={"accuracy": 0.89, "f1_score": 0.85, "loss": 0.32},
        )
        comparisons = compare_metrics(regressed, self.baseline, tolerance=0.02)
        acc = next(c for c in comparisons if c.name == "accuracy")
        assert acc.regression is True
        loss = next(c for c in comparisons if c.name == "loss")
        assert loss.regression is True  # loss increased significantly

    def test_within_tolerance(self):
        # Tiny change that should be within default 2% tolerance
        slight = MetricsData(
            model_name="slight",
            metrics={"accuracy": 0.929, "f1_score": 0.909, "loss": 0.181},
        )
        comparisons = compare_metrics(slight, self.baseline, tolerance=0.02)
        for comp in comparisons:
            assert comp.regression is False

    def test_only_shared_metrics(self):
        current = MetricsData(model_name="a", metrics={"accuracy": 0.9, "new_metric": 0.5})
        baseline = MetricsData(model_name="b", metrics={"accuracy": 0.88, "old_metric": 0.6})
        comparisons = compare_metrics(current, baseline)
        assert len(comparisons) == 1
        assert comparisons[0].name == "accuracy"

    def test_identical_metrics(self):
        comparisons = compare_metrics(self.baseline, self.baseline)
        for comp in comparisons:
            assert comp.delta == 0
            assert comp.regression is False

    def test_zero_baseline_value(self):
        current = MetricsData(model_name="a", metrics={"accuracy": 0.5})
        baseline = MetricsData(model_name="b", metrics={"accuracy": 0.0})
        comparisons = compare_metrics(current, baseline)
        assert len(comparisons) == 1
        assert comparisons[0].delta_pct == float("inf")

    def test_custom_higher_is_better(self):
        current = MetricsData(model_name="a", metrics={"custom": 0.5})
        baseline = MetricsData(model_name="b", metrics={"custom": 0.6})
        # Default: higher is better, so this is a regression
        comps = compare_metrics(current, baseline, tolerance=0.01)
        assert comps[0].regression is True
        # Override: lower is better, so this is an improvement
        comps = compare_metrics(current, baseline, tolerance=0.01, higher_is_better={"custom": False})
        assert comps[0].improved is True
        assert comps[0].regression is False

    def test_per_metric_tolerance_override(self):
        current = MetricsData(model_name="a", metrics={"accuracy": 0.91})
        baseline = MetricsData(model_name="b", metrics={"accuracy": 0.93})

        comps = compare_metrics(
            current,
            baseline,
            tolerance=0.01,
            metric_tolerances={"accuracy": 0.05},
        )

        assert comps[0].tolerance == 0.05
        assert comps[0].regression is False

    def test_per_metric_severity_override(self):
        current = MetricsData(model_name="a", metrics={"accuracy": 0.89})
        baseline = MetricsData(model_name="b", metrics={"accuracy": 0.93})

        comps = compare_metrics(
            current,
            baseline,
            tolerance=0.01,
            metric_severities={"accuracy": "warn"},
        )

        assert comps[0].severity == "warn"
        assert comps[0].regression is True


class TestLoadMetricsWithObservations:
    def test_load_file_with_observations(self):
        path = os.path.join(FIXTURES_DIR, "current_metrics_with_observations.json")
        data = load_metrics(path)
        assert "accuracy" in data.observations
        assert len(data.observations["accuracy"]) == 5
        assert all(isinstance(v, (int, float)) for v in data.observations["accuracy"])

    def test_load_file_without_observations(self):
        path = os.path.join(FIXTURES_DIR, "current_metrics.json")
        data = load_metrics(path)
        assert data.observations == {}

    def test_observations_non_numeric_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "metrics": {"accuracy": 0.9},
                "observations": {"accuracy": [0.9, "bad", 0.8]},
            }, f)
            f.flush()
            with pytest.raises(ValueError, match="numeric"):
                load_metrics(f.name)
        os.unlink(f.name)

    def test_observations_non_list_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "metrics": {"accuracy": 0.9},
                "observations": {"accuracy": 0.9},
            }, f)
            f.flush()
            with pytest.raises(ValueError, match="list"):
                load_metrics(f.name)
        os.unlink(f.name)


class TestValidatePairedObservations:
    def test_valid_paired_observations(self):
        current = MetricsData(
            model_name="a",
            metrics={"accuracy": 0.95},
            observations={"accuracy": [0.94, 0.95, 0.96]},
        )
        baseline = MetricsData(
            model_name="b",
            metrics={"accuracy": 0.93},
            observations={"accuracy": [0.92, 0.93, 0.94]},
        )
        pairs = validate_paired_observations(current, baseline, ["accuracy"])
        assert "accuracy" in pairs
        assert len(pairs["accuracy"][0]) == 3
        assert len(pairs["accuracy"][1]) == 3

    def test_missing_current_observations_raises(self):
        current = MetricsData(model_name="a", metrics={"accuracy": 0.95})
        baseline = MetricsData(
            model_name="b",
            metrics={"accuracy": 0.93},
            observations={"accuracy": [0.92, 0.93, 0.94]},
        )
        with pytest.raises(ValueError, match="Current metrics file has no 'observations'"):
            validate_paired_observations(current, baseline, ["accuracy"])

    def test_missing_baseline_observations_raises(self):
        current = MetricsData(
            model_name="a",
            metrics={"accuracy": 0.95},
            observations={"accuracy": [0.94, 0.95, 0.96]},
        )
        baseline = MetricsData(model_name="b", metrics={"accuracy": 0.93})
        with pytest.raises(ValueError, match="Baseline metrics file has no 'observations'"):
            validate_paired_observations(current, baseline, ["accuracy"])

    def test_mismatched_lengths_raises(self):
        current = MetricsData(
            model_name="a",
            metrics={"accuracy": 0.95},
            observations={"accuracy": [0.94, 0.95, 0.96]},
        )
        baseline = MetricsData(
            model_name="b",
            metrics={"accuracy": 0.93},
            observations={"accuracy": [0.92, 0.93]},
        )
        with pytest.raises(ValueError, match="mismatch"):
            validate_paired_observations(current, baseline, ["accuracy"])

    def test_missing_metric_in_observations_raises(self):
        current = MetricsData(
            model_name="a",
            metrics={"accuracy": 0.95, "f1": 0.9},
            observations={"accuracy": [0.94, 0.95, 0.96]},
        )
        baseline = MetricsData(
            model_name="b",
            metrics={"accuracy": 0.93, "f1": 0.88},
            observations={"accuracy": [0.92, 0.93, 0.94], "f1": [0.87, 0.88, 0.89]},
        )
        with pytest.raises(ValueError, match="no observations for 'f1'"):
            validate_paired_observations(current, baseline, ["accuracy", "f1"])

    def test_too_few_observations_raises(self):
        current = MetricsData(
            model_name="a",
            metrics={"accuracy": 0.95},
            observations={"accuracy": [0.95]},
        )
        baseline = MetricsData(
            model_name="b",
            metrics={"accuracy": 0.93},
            observations={"accuracy": [0.93]},
        )
        with pytest.raises(ValueError, match="At least 2"):
            validate_paired_observations(current, baseline, ["accuracy"])
