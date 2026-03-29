"""Tests for src.reporters.pr_comment."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from src.reporters.pr_comment import (
    COMMENT_MARKER,
    PROJECT_URL,
    _build_metrics_table,
    _build_regression_summary,
    _build_statistical_details,
    _fmt_delta,
    _fmt_metric,
    generate_report,
    get_pr_number,
    post_or_update_comment,
)
from src.utils.metrics import MetricComparison
from src.utils.stats import RegressionResult
from src.validators.model_validator import ModelValidationResult


def _make_model_result(regression: bool = False) -> ModelValidationResult:
    comparisons = [
        MetricComparison(
            name="accuracy", current=0.95, baseline=0.93,
            delta=0.02, delta_pct=0.0215, higher_is_better=True,
            improved=True, regression=False,
        ),
        MetricComparison(
            name="f1_score", current=0.93, baseline=0.91,
            delta=0.02, delta_pct=0.022, higher_is_better=True,
            improved=True, regression=False,
        ),
        MetricComparison(
            name="loss", current=0.15, baseline=0.18,
            delta=-0.03, delta_pct=-0.167, higher_is_better=False,
            improved=True, regression=False,
        ),
    ]
    if regression:
        comparisons[0] = MetricComparison(
            name="accuracy", current=0.89, baseline=0.93,
            delta=-0.04, delta_pct=-0.043, higher_is_better=True,
            improved=False, regression=True,
        )
    return ModelValidationResult(
        comparisons=comparisons,
        regression_result=RegressionResult(
            method="threshold",
            regression_detected=regression,
            details={"tolerance": 0.02, "total_metrics": 3, "regressed_count": 1 if regression else 0, "regressed_metrics": []},
        ),
        model_name="test-model",
        framework="pytorch",
        summary="All 3 metrics within tolerance" if not regression else "REGRESSION DETECTED: 1 of 3 metrics degraded",
    )


class TestGenerateReport:
    def test_passing_report(self):
        result = _make_model_result(regression=False)
        report = generate_report(model_result=result)
        assert COMMENT_MARKER in report
        assert ":white_check_mark:" in report
        assert "accuracy" in report
        assert "f1_score" in report
        assert "loss" in report
        assert "ML-CI" in report
        assert PROJECT_URL in report
        assert "github.com/ml-ci/ml-ci-action" not in report

    def test_failing_report(self):
        result = _make_model_result(regression=True)
        report = generate_report(model_result=result)
        assert COMMENT_MARKER in report
        assert ":x:" in report
        assert "REGRESSION" in report or "regressed" in report

    def test_report_with_model_card(self):
        result = _make_model_result()
        report = generate_report(model_result=result, model_card_path="MODEL_CARD.md")
        assert "MODEL_CARD.md" in report
        assert "Model Card" in report

    def test_report_without_model_result(self):
        report = generate_report()
        assert COMMENT_MARKER in report
        assert "ML-CI" in report

    def test_current_only_report_contains_current_metrics_table(self):
        report = generate_report(
            current_metrics={"accuracy": 0.95, "loss": 0.12},
            current_only_metrics=["accuracy", "loss"],
            baseline_source={"mode": "none"},
        )
        assert "Current Metrics" in report
        assert "| Metric | Current |" in report
        assert "No baseline available" in report
        assert "Metric Coverage" in report

    def test_report_contains_marker_for_idempotent_updates(self):
        report = generate_report(model_result=_make_model_result())
        assert report.startswith(COMMENT_MARKER)

    def test_metric_coverage_section_lists_non_shared_metrics(self):
        result = _make_model_result()
        result.current_only_metrics = ["precision"]
        result.baseline_only_metrics = ["recall"]

        report = generate_report(model_result=result)

        assert "Metric Coverage" in report
        assert "Current-only metrics" in report
        assert "Baseline-only metrics" in report


class TestBuildMetricsTable:
    def test_table_structure(self):
        comparisons = [
            MetricComparison(
                name="accuracy", current=0.95, baseline=0.93,
                delta=0.02, delta_pct=0.0215, higher_is_better=True,
                improved=True, regression=False,
            ),
        ]
        table = _build_metrics_table(comparisons)
        assert "| Metric |" in table
        assert "accuracy" in table
        assert "0.9500" in table
        assert "0.9300" in table

    def test_table_status_icons(self):
        comparisons = [
            MetricComparison(
                name="good", current=0.95, baseline=0.93,
                delta=0.02, delta_pct=0.0215, higher_is_better=True,
                improved=True, regression=False,
            ),
            MetricComparison(
                name="bad", current=0.89, baseline=0.93,
                delta=-0.04, delta_pct=-0.043, higher_is_better=True,
                improved=False, regression=True,
            ),
        ]
        table = _build_metrics_table(comparisons)
        assert ":white_check_mark:" in table  # good metric
        assert ":x:" in table  # regressed metric


class TestFmtMetric:
    def test_normal_value(self):
        assert _fmt_metric(0.9500) == "0.9500"

    def test_large_value(self):
        assert "," in _fmt_metric(10000.5)

    def test_small_value(self):
        assert "e" in _fmt_metric(0.0001)

    def test_zero(self):
        assert _fmt_metric(0.0) == "0.0000"


class TestFmtDelta:
    def test_positive(self):
        assert _fmt_delta(0.02).startswith("+")

    def test_negative(self):
        assert _fmt_delta(-0.02).startswith("-")

    def test_zero(self):
        assert _fmt_delta(0.0) == "0"


class TestGetPrNumber:
    def test_pull_request_event(self):
        event = {"pull_request": {"number": 42}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(event, f)
            f.flush()
            os.environ["GITHUB_EVENT_PATH"] = f.name
            assert get_pr_number() == 42
        os.unlink(f.name)
        del os.environ["GITHUB_EVENT_PATH"]

    def test_no_event_path(self):
        os.environ.pop("GITHUB_EVENT_PATH", None)
        assert get_pr_number() is None

    def test_non_pr_event(self):
        event = {"ref": "refs/heads/main", "before": "abc123"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(event, f)
            f.flush()
            os.environ["GITHUB_EVENT_PATH"] = f.name
            assert get_pr_number() is None
        os.unlink(f.name)
        del os.environ["GITHUB_EVENT_PATH"]


class TestPostOrUpdateComment:
    @patch("src.reporters.pr_comment.requests")
    def test_creates_new_comment(self, mock_requests):
        # No existing comments
        mock_get_resp = MagicMock()
        mock_get_resp.json.return_value = []
        mock_get_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_get_resp

        mock_post_resp = MagicMock()
        mock_post_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_post_resp

        post_or_update_comment("owner/repo", 1, "test body", "token123")
        mock_requests.post.assert_called_once()

    @patch("src.reporters.pr_comment.requests")
    def test_updates_existing_comment(self, mock_requests):
        # Existing comment with marker
        mock_get_resp = MagicMock()
        mock_get_resp.json.return_value = [
            {"id": 999, "body": f"{COMMENT_MARKER}\nold report"},
        ]
        mock_get_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_get_resp

        mock_patch_resp = MagicMock()
        mock_patch_resp.raise_for_status = MagicMock()
        mock_requests.patch.return_value = mock_patch_resp

        post_or_update_comment("owner/repo", 1, "new body", "token123")
        mock_requests.patch.assert_called_once()
        # Should NOT create a new comment
        mock_requests.post.assert_not_called()


class TestStatisticalReportRendering:
    def test_wilcoxon_report_contains_p_values(self):
        rr = RegressionResult(
            method="wilcoxon",
            regression_detected=False,
            details={
                "alpha": 0.05,
                "total_metrics": 1,
                "regressed_count": 0,
                "regressed_metrics": [],
                "metric_results": {
                    "accuracy": {
                        "p_value": 0.0625,
                        "statistic": 0.0,
                        "median_diff": 0.02,
                        "n_observations": 5,
                        "significant": False,
                        "regressed": False,
                    },
                },
            },
        )
        details = _build_statistical_details(rr)
        assert details is not None
        assert "Wilcoxon" in details
        assert "p-value" in details
        assert "0.0625" in details
        assert "<details>" in details

    def test_bootstrap_report_contains_confidence_interval(self):
        rr = RegressionResult(
            method="bootstrap",
            regression_detected=False,
            details={
                "confidence": 0.95,
                "n_bootstrap": 10000,
                "total_metrics": 1,
                "regressed_count": 0,
                "regressed_metrics": [],
                "metric_results": {
                    "accuracy": {
                        "ci_lower": 0.015,
                        "ci_upper": 0.025,
                        "mean_diff": 0.02,
                        "n_observations": 5,
                        "n_bootstrap": 10000,
                        "regressed": False,
                    },
                },
            },
        )
        details = _build_statistical_details(rr)
        assert details is not None
        assert "Bootstrap" in details
        assert "CI Lower" in details
        assert "CI Upper" in details
        assert "<details>" in details

    def test_threshold_returns_no_statistical_details(self):
        rr = RegressionResult(
            method="threshold",
            regression_detected=False,
            details={"tolerance": 0.02, "total_metrics": 1, "regressed_count": 0, "regressed_metrics": []},
        )
        assert _build_statistical_details(rr) is None

    def test_wilcoxon_summary_shows_alpha(self):
        rr = RegressionResult(
            method="wilcoxon",
            regression_detected=False,
            details={
                "alpha": 0.05,
                "total_metrics": 1,
                "regressed_count": 0,
                "regressed_metrics": [],
                "metric_results": {
                    "accuracy": {
                        "p_value": 0.5,
                        "statistic": 0.0,
                        "median_diff": 0.0,
                        "n_observations": 5,
                        "significant": False,
                        "regressed": False,
                    },
                },
            },
        )
        summary = _build_regression_summary(rr)
        assert "wilcoxon" in summary
        assert "alpha" in summary
        assert "No statistically significant regression" in summary

    def test_bootstrap_summary_shows_confidence(self):
        rr = RegressionResult(
            method="bootstrap",
            regression_detected=True,
            details={
                "confidence": 0.95,
                "n_bootstrap": 10000,
                "total_metrics": 1,
                "regressed_count": 1,
                "regressed_metrics": [{"metric": "accuracy"}],
                "metric_results": {},
            },
        )
        summary = _build_regression_summary(rr)
        assert "bootstrap" in summary
        assert "95%" in summary
        assert "1 blocking metric(s) regressed" in summary
