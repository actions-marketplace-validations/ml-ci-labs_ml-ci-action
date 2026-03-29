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
    current_only_metrics: list[str] = field(default_factory=list)
    baseline_only_metrics: list[str] = field(default_factory=list)


def validate_model(
    current: MetricsData,
    baseline: MetricsData,
    regression_method: str = "threshold",
    tolerance: float = 0.02,
    higher_is_better: dict[str, bool] | None = None,
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
        alpha: Significance level for Wilcoxon test.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level for bootstrap CI.

    Returns:
        ModelValidationResult with comparisons, regression result, and summary.
    """
    comparisons = compare_metrics(current, baseline, tolerance, higher_is_better)

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

    # Identify metrics only in one side
    current_keys = set(current.metrics.keys())
    baseline_keys = set(baseline.metrics.keys())
    current_only = sorted(current_keys - baseline_keys)
    baseline_only = sorted(baseline_keys - current_keys)

    # Build summary
    n_compared = len(comparisons)
    n_regressed = regression_result.details.get("regressed_count", 0)

    if n_compared == 0:
        summary = "No shared metrics between current and baseline — skipped regression detection"
    elif regression_result.regression_detected:
        if regression_method == "threshold":
            summary = (
                f"REGRESSION DETECTED: {n_regressed} of {n_compared} metrics "
                f"degraded beyond {tolerance:.1%} tolerance"
            )
        elif regression_method == "wilcoxon":
            summary = (
                f"REGRESSION DETECTED: {n_regressed} of {n_compared} metrics "
                f"showed statistically significant degradation (alpha={alpha})"
            )
        else:
            summary = (
                f"REGRESSION DETECTED: {n_regressed} of {n_compared} metrics "
                f"showed degradation ({confidence:.0%} CI entirely on bad side)"
            )
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
        current_only_metrics=current_only,
        baseline_only_metrics=baseline_only,
    )
