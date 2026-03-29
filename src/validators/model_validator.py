"""Model performance validation and regression detection."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.utils.metrics import (
    MetricComparison,
    MetricsData,
    _is_higher_better,
    compare_metrics,
    validate_paired_observations,
)
from src.utils.stats import RegressionResult, bootstrap_ci, threshold_test, wilcoxon_test


@dataclass
class ModelValidationResult:
    """Full result of model validation."""

    comparisons: list[MetricComparison]
    regression_result: RegressionResult
    model_name: str
    framework: str
    summary: str  # human-readable one-liner
    blocking_regression_count: int = 0
    warning_regression_count: int = 0
    current_only_metrics: list[str] = field(default_factory=list)
    baseline_only_metrics: list[str] = field(default_factory=list)


def validate_model(
    current: MetricsData,
    baseline: MetricsData,
    regression_method: str = "threshold",
    tolerance: float = 0.02,
    higher_is_better: dict[str, bool] | None = None,
    metric_tolerances: dict[str, float] | None = None,
    metric_severities: dict[str, str] | None = None,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> ModelValidationResult:
    """Run full model validation: compare metrics and detect regressions.

    Args:
        current: Current model metrics from the PR.
        baseline: Baseline model metrics (e.g., from main branch).
        regression_method: Detection method ("threshold", "wilcoxon", "bootstrap").
        tolerance: Maximum allowed degradation fraction (threshold method).
        higher_is_better: Optional direction overrides per metric.
        metric_tolerances: Optional per-metric threshold overrides.
        metric_severities: Optional per-metric severity overrides.
        alpha: Significance level for Wilcoxon test.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level for bootstrap CI.

    Returns:
        ModelValidationResult with comparisons, regression result, and summary.
    """
    comparisons = compare_metrics(
        current,
        baseline,
        tolerance=tolerance,
        higher_is_better=higher_is_better,
        metric_tolerances=metric_tolerances,
        metric_severities=metric_severities,
    )

    # Build direction dict for statistical tests
    overrides = higher_is_better or {}
    shared_metrics = sorted(set(current.metrics.keys()) & set(baseline.metrics.keys()))
    direction_map = {name: overrides.get(name, _is_higher_better(name)) for name in shared_metrics}

    if regression_method == "threshold":
        regression_result = threshold_test(comparisons, tolerance)
    elif regression_method == "wilcoxon":
        obs_pairs = validate_paired_observations(current, baseline, shared_metrics)
        regression_result = wilcoxon_test(obs_pairs, direction_map, alpha=alpha)
    elif regression_method == "bootstrap":
        obs_pairs = validate_paired_observations(current, baseline, shared_metrics)
        regression_result = bootstrap_ci(
            obs_pairs, direction_map, n_bootstrap=n_bootstrap, confidence=confidence,
        )
    else:
        raise ValueError(
            f"Unknown regression method: '{regression_method}'. "
            "Supported methods: 'threshold', 'wilcoxon', 'bootstrap'."
        )

    _apply_effective_regressions(comparisons, regression_result)

    # Identify metrics only in one side
    current_keys = set(current.metrics.keys())
    baseline_keys = set(baseline.metrics.keys())
    current_only = sorted(current_keys - baseline_keys)
    baseline_only = sorted(baseline_keys - current_keys)

    # Build summary
    n_compared = len(comparisons)
    n_regressed = regression_result.details.get("regressed_count", 0)
    n_blocking = regression_result.details.get("blocking_regressed_count", 0)
    n_warning = regression_result.details.get("warning_regressed_count", 0)

    if n_compared == 0:
        summary = "No shared metrics between current and baseline — skipped regression detection"
    elif n_regressed:
        if regression_method == "threshold":
            criterion = "degraded beyond configured thresholds"
        elif regression_method == "wilcoxon":
            criterion = f"showed statistically significant degradation (alpha={alpha})"
        else:
            criterion = f"showed degradation ({confidence:.0%} CI entirely on bad side)"

        if n_blocking and n_warning:
            summary = (
                f"REGRESSION DETECTED: {n_blocking} blocking and {n_warning} warning "
                f"metric(s) {criterion}"
            )
        elif n_blocking:
            summary = f"REGRESSION DETECTED: {n_blocking} blocking metric(s) {criterion}"
        else:
            summary = f"Warnings only: {n_warning} metric(s) {criterion}"
    else:
        n_improved = sum(1 for c in comparisons if c.improved and c.delta != 0)
        n_unchanged = n_compared - n_improved - n_regressed
        parts = []
        if n_improved:
            parts.append(f"{n_improved} improved")
        if n_unchanged:
            parts.append(f"{n_unchanged} stable")
        if regression_method == "threshold":
            summary = f"All {n_compared} metrics within tolerance ({', '.join(parts)})"
        elif regression_method == "wilcoxon":
            summary = f"All {n_compared} metrics passed Wilcoxon test ({', '.join(parts)})"
        else:
            summary = f"All {n_compared} metrics passed bootstrap CI test ({', '.join(parts)})"

    return ModelValidationResult(
        comparisons=comparisons,
        regression_result=regression_result,
        model_name=current.model_name,
        framework=current.framework,
        summary=summary,
        blocking_regression_count=n_blocking,
        warning_regression_count=n_warning,
        current_only_metrics=current_only,
        baseline_only_metrics=baseline_only,
    )


def _apply_effective_regressions(
    comparisons: list[MetricComparison],
    regression_result: RegressionResult,
) -> None:
    """Align per-metric regression state and severity counts."""
    metric_results = regression_result.details.get("metric_results")

    if metric_results:
        for comparison in comparisons:
            comparison.regression = bool(metric_results.get(comparison.name, {}).get("regressed", False))

    regressed_metrics: list[dict[str, object]] = []
    blocking = 0
    warning = 0

    for comparison in comparisons:
        if not comparison.regression:
            continue
        if comparison.severity == "warn":
            warning += 1
        else:
            blocking += 1

        entry: dict[str, object] = {
            "metric": comparison.name,
            "current": comparison.current,
            "baseline": comparison.baseline,
            "delta": comparison.delta,
            "delta_pct": comparison.delta_pct,
            "higher_is_better": comparison.higher_is_better,
            "severity": comparison.severity,
        }
        if metric_results and comparison.name in metric_results:
            metric_results[comparison.name]["severity"] = comparison.severity
            entry.update(metric_results[comparison.name])
        else:
            entry["tolerance"] = comparison.tolerance
        regressed_metrics.append(entry)

    regression_result.regression_detected = bool(regressed_metrics)
    regression_result.details["regressed_count"] = len(regressed_metrics)
    regression_result.details["blocking_regressed_count"] = blocking
    regression_result.details["warning_regressed_count"] = warning
    regression_result.details["regressed_metrics"] = regressed_metrics
