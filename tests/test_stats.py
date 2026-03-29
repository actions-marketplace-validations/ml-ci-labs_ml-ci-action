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


# --- Wilcoxon tests ---


class TestWilcoxonTest:
    def test_no_regression_when_current_better(self):
        obs = {
            "accuracy": ([0.95, 0.96, 0.94, 0.95, 0.96], [0.93, 0.94, 0.92, 0.93, 0.94]),
        }
        result = wilcoxon_test(obs, {"accuracy": True})
        assert result.method == "wilcoxon"
        assert result.regression_detected is False

    def test_regression_detected_when_current_worse(self):
        # Need >= 6 observations for Wilcoxon to reach p < 0.05
        obs = {
            "accuracy": (
                [0.88, 0.89, 0.87, 0.88, 0.89, 0.88, 0.87, 0.89, 0.88, 0.87],
                [0.93, 0.94, 0.92, 0.93, 0.94, 0.93, 0.92, 0.94, 0.93, 0.92],
            ),
        }
        result = wilcoxon_test(obs, {"accuracy": True})
        assert result.regression_detected is True
        assert result.details["regressed_count"] == 1
        assert result.details["metric_results"]["accuracy"]["regressed"] is True

    def test_not_significant_small_difference(self):
        # Very small, noisy differences shouldn't be significant
        obs = {
            "accuracy": ([0.930, 0.941, 0.919, 0.931, 0.940],
                         [0.930, 0.940, 0.920, 0.930, 0.940]),
        }
        result = wilcoxon_test(obs, {"accuracy": True})
        assert result.regression_detected is False

    def test_single_metric_regression_flags_overall(self):
        # Need >= 6 observations for significance
        obs = {
            "accuracy": (
                [0.88, 0.89, 0.87, 0.88, 0.89, 0.88, 0.87, 0.89, 0.88, 0.87],
                [0.93, 0.94, 0.92, 0.93, 0.94, 0.93, 0.92, 0.94, 0.93, 0.92],
            ),
            "f1": (
                [0.95, 0.96, 0.94, 0.95, 0.96, 0.95, 0.94, 0.96, 0.95, 0.94],
                [0.89, 0.90, 0.88, 0.89, 0.90, 0.89, 0.88, 0.90, 0.89, 0.88],
            ),
        }
        result = wilcoxon_test(obs, {"accuracy": True, "f1": True})
        assert result.regression_detected is True
        assert result.details["regressed_count"] == 1

    def test_lower_is_better_direction(self):
        # Loss increased -> regression for lower-is-better (need >= 6 obs)
        obs = {
            "loss": (
                [0.30, 0.29, 0.31, 0.30, 0.29, 0.30, 0.31, 0.29, 0.30, 0.31],
                [0.18, 0.17, 0.19, 0.18, 0.17, 0.18, 0.19, 0.17, 0.18, 0.19],
            ),
        }
        result = wilcoxon_test(obs, {"loss": False})
        assert result.regression_detected is True

    def test_lower_is_better_improvement(self):
        # Loss decreased -> no regression for lower-is-better
        obs = {
            "loss": ([0.15, 0.14, 0.16, 0.15, 0.14], [0.18, 0.17, 0.19, 0.18, 0.17]),
        }
        result = wilcoxon_test(obs, {"loss": False})
        assert result.regression_detected is False

    def test_identical_observations_no_regression(self):
        obs = {
            "accuracy": ([0.93, 0.94, 0.92, 0.93, 0.94], [0.93, 0.94, 0.92, 0.93, 0.94]),
        }
        result = wilcoxon_test(obs, {"accuracy": True})
        assert result.regression_detected is False
        assert result.details["metric_results"]["accuracy"]["p_value"] == 1.0

    def test_custom_alpha(self):
        # With very strict alpha, even real differences may not be significant
        obs = {
            "accuracy": ([0.88, 0.89, 0.87, 0.88, 0.89], [0.93, 0.94, 0.92, 0.93, 0.94]),
        }
        result = wilcoxon_test(obs, {"accuracy": True}, alpha=0.001)
        # With only 5 observations, Wilcoxon p-value minimum is ~0.0625, so alpha=0.001 won't flag
        assert result.regression_detected is False

    def test_details_contain_expected_keys(self):
        obs = {
            "accuracy": ([0.95, 0.96, 0.94, 0.95, 0.96], [0.93, 0.94, 0.92, 0.93, 0.94]),
        }
        result = wilcoxon_test(obs, {"accuracy": True})
        assert "alpha" in result.details
        assert "total_metrics" in result.details
        assert "regressed_count" in result.details
        assert "regressed_metrics" in result.details
        assert "metric_results" in result.details
        mr = result.details["metric_results"]["accuracy"]
        assert "p_value" in mr
        assert "statistic" in mr
        assert "median_diff" in mr
        assert "n_observations" in mr
        assert "significant" in mr
        assert "regressed" in mr

    def test_multiple_metrics_all_pass(self):
        obs = {
            "accuracy": ([0.95, 0.96, 0.94, 0.95, 0.96], [0.93, 0.94, 0.92, 0.93, 0.94]),
            "f1": ([0.91, 0.92, 0.90, 0.91, 0.92], [0.89, 0.90, 0.88, 0.89, 0.90]),
        }
        result = wilcoxon_test(obs, {"accuracy": True, "f1": True})
        assert result.regression_detected is False
        assert result.details["total_metrics"] == 2


# --- Bootstrap CI tests ---


class TestBootstrapCI:
    def test_no_regression_improvement(self):
        obs = {
            "accuracy": ([0.95, 0.96, 0.94, 0.95, 0.96], [0.93, 0.94, 0.92, 0.93, 0.94]),
        }
        result = bootstrap_ci(obs, {"accuracy": True}, seed=42)
        assert result.method == "bootstrap"
        assert result.regression_detected is False

    def test_regression_detected(self):
        obs = {
            "accuracy": ([0.88, 0.89, 0.87, 0.88, 0.89], [0.93, 0.94, 0.92, 0.93, 0.94]),
        }
        result = bootstrap_ci(obs, {"accuracy": True}, seed=42)
        assert result.regression_detected is True
        assert result.details["regressed_count"] == 1

    def test_ci_spans_zero_no_regression(self):
        # Very noisy data where CI should span zero
        obs = {
            "accuracy": ([0.93, 0.95, 0.91, 0.94, 0.92], [0.93, 0.94, 0.92, 0.93, 0.94]),
        }
        result = bootstrap_ci(obs, {"accuracy": True}, seed=42)
        mr = result.details["metric_results"]["accuracy"]
        # CI spans zero or is positive -> not regressed
        assert mr["regressed"] is False

    def test_lower_is_better_metric(self):
        # Loss increased -> regression
        obs = {
            "loss": ([0.30, 0.29, 0.31, 0.30, 0.29], [0.18, 0.17, 0.19, 0.18, 0.17]),
        }
        result = bootstrap_ci(obs, {"loss": False}, seed=42)
        assert result.regression_detected is True

    def test_lower_is_better_improvement(self):
        # Loss decreased -> no regression
        obs = {
            "loss": ([0.15, 0.14, 0.16, 0.15, 0.14], [0.18, 0.17, 0.19, 0.18, 0.17]),
        }
        result = bootstrap_ci(obs, {"loss": False}, seed=42)
        assert result.regression_detected is False

    def test_reproducibility_with_seed(self):
        obs = {
            "accuracy": ([0.95, 0.96, 0.94, 0.95, 0.96], [0.93, 0.94, 0.92, 0.93, 0.94]),
        }
        r1 = bootstrap_ci(obs, {"accuracy": True}, seed=123)
        r2 = bootstrap_ci(obs, {"accuracy": True}, seed=123)
        ci1 = r1.details["metric_results"]["accuracy"]
        ci2 = r2.details["metric_results"]["accuracy"]
        assert ci1["ci_lower"] == ci2["ci_lower"]
        assert ci1["ci_upper"] == ci2["ci_upper"]

    def test_custom_confidence_level(self):
        obs = {
            "accuracy": ([0.95, 0.96, 0.94, 0.95, 0.96], [0.93, 0.94, 0.92, 0.93, 0.94]),
        }
        r_90 = bootstrap_ci(obs, {"accuracy": True}, confidence=0.90, seed=42)
        r_99 = bootstrap_ci(obs, {"accuracy": True}, confidence=0.99, seed=42)
        # 99% CI should be wider than 90% CI
        ci_90 = r_90.details["metric_results"]["accuracy"]
        ci_99 = r_99.details["metric_results"]["accuracy"]
        width_90 = ci_90["ci_upper"] - ci_90["ci_lower"]
        width_99 = ci_99["ci_upper"] - ci_99["ci_lower"]
        assert width_99 > width_90

    def test_custom_n_bootstrap(self):
        obs = {
            "accuracy": ([0.95, 0.96, 0.94, 0.95, 0.96], [0.93, 0.94, 0.92, 0.93, 0.94]),
        }
        result = bootstrap_ci(obs, {"accuracy": True}, n_bootstrap=500, seed=42)
        assert result.details["n_bootstrap"] == 500

    def test_details_contain_expected_keys(self):
        obs = {
            "accuracy": ([0.95, 0.96, 0.94, 0.95, 0.96], [0.93, 0.94, 0.92, 0.93, 0.94]),
        }
        result = bootstrap_ci(obs, {"accuracy": True}, seed=42)
        assert "confidence" in result.details
        assert "n_bootstrap" in result.details
        assert "total_metrics" in result.details
        assert "regressed_count" in result.details
        assert "regressed_metrics" in result.details
        assert "metric_results" in result.details
        mr = result.details["metric_results"]["accuracy"]
        assert "ci_lower" in mr
        assert "ci_upper" in mr
        assert "mean_diff" in mr
        assert "n_observations" in mr
        assert "regressed" in mr

    def test_single_metric_regression(self):
        obs = {
            "accuracy": ([0.88, 0.89, 0.87, 0.88, 0.89], [0.93, 0.94, 0.92, 0.93, 0.94]),
        }
        result = bootstrap_ci(obs, {"accuracy": True}, seed=42)
        mr = result.details["metric_results"]["accuracy"]
        assert mr["ci_upper"] < 0  # entire CI below zero
        assert mr["regressed"] is True
