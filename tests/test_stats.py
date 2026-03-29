"""Tests for src.utils.stats."""

import pytest

from src.utils.metrics import MetricComparison
from src.utils.stats import bootstrap_ci, threshold_test, wilcoxon_test


def _make_comparison(
    name: str = "accuracy",
    current: float = 0.95,
    baseline: float = 0.93,
    regression: bool = False,
    higher_is_better: bool = True,
) -> MetricComparison:
    delta = current - baseline
    delta_pct = delta / abs(baseline) if baseline != 0 else 0.0
    improved = (delta >= 0) if higher_is_better else (delta <= 0)
    return MetricComparison(
        name=name,
        current=current,
        baseline=baseline,
        delta=delta,
        delta_pct=delta_pct,
        higher_is_better=higher_is_better,
        improved=improved,
        regression=regression,
    )


class TestThresholdTest:
    def test_no_regression(self):
        comparisons = [
            _make_comparison("accuracy", 0.95, 0.93, regression=False),
            _make_comparison("f1", 0.93, 0.91, regression=False),
        ]
        result = threshold_test(comparisons)
        assert result.method == "threshold"
        assert result.regression_detected is False
        assert result.details["regressed_count"] == 0

    def test_regression_detected(self):
        comparisons = [
            _make_comparison("accuracy", 0.89, 0.93, regression=True),
            _make_comparison("f1", 0.93, 0.91, regression=False),
        ]
        result = threshold_test(comparisons)
        assert result.regression_detected is True
        assert result.details["regressed_count"] == 1
        assert result.details["regressed_metrics"][0]["metric"] == "accuracy"

    def test_multiple_regressions(self):
        comparisons = [
            _make_comparison("accuracy", 0.80, 0.93, regression=True),
            _make_comparison("f1", 0.75, 0.91, regression=True),
            _make_comparison("loss", 0.50, 0.18, regression=True, higher_is_better=False),
        ]
        result = threshold_test(comparisons)
        assert result.regression_detected is True
        assert result.details["regressed_count"] == 3

    def test_empty_comparisons(self):
        result = threshold_test([])
        assert result.regression_detected is False
        assert result.details["total_metrics"] == 0

    def test_loss_metric_direction(self):
        comparisons = [
            _make_comparison("loss", 0.20, 0.18, regression=False, higher_is_better=False),
        ]
        result = threshold_test(comparisons)
        assert result.regression_detected is False

    def test_all_details_present(self):
        comparisons = [_make_comparison("accuracy", 0.95, 0.93, regression=False)]
        result = threshold_test(comparisons)
        assert "tolerance" in result.details
        assert "total_metrics" in result.details
        assert "regressed_count" in result.details
        assert "regressed_metrics" in result.details


class TestWilcoxonTest:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="v0.2"):
            wilcoxon_test([0.9, 0.91], [0.88, 0.89])


class TestBootstrapCI:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="v0.2"):
            bootstrap_ci([0.9, 0.91], [0.88, 0.89])
