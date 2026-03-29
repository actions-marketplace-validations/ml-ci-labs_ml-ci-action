"""Model performance validation and regression detection."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.utils.metrics import MetricComparison, MetricsData, compare_metrics
from src.utils.stats import RegressionResult, threshold_test


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
) -> ModelValidationResult:
    """Run full model validation: compare metrics and detect regressions.

    Args:
        current: Current model metrics from the PR.
        baseline: Baseline model metrics (e.g., from main branch).
        regression_method: Detection method ("threshold" for v0.1.1).
        tolerance: Maximum allowed degradation fraction.
        higher_is_better: Optional direction overrides per metric.

    Returns:
        ModelValidationResult with comparisons, regression result, and summary.
    """
    comparisons = compare_metrics(current, baseline, tolerance, higher_is_better)

    if regression_method == "threshold":
        regression_result = threshold_test(comparisons, tolerance)
    else:
        raise ValueError(
            f"Unknown regression method: '{regression_method}'. "
            "Supported methods: 'threshold'. "
            "'wilcoxon' and 'bootstrap' will be available in v0.2."
        )

    # Identify metrics only in one side
    current_keys = set(current.metrics.keys())
    baseline_keys = set(baseline.metrics.keys())
    current_only = sorted(current_keys - baseline_keys)
    baseline_only = sorted(baseline_keys - current_keys)

    # Build summary
    n_compared = len(comparisons)
    n_improved = sum(1 for c in comparisons if c.improved and c.delta != 0)
    n_regressed = regression_result.details.get("regressed_count", 0)
    n_unchanged = n_compared - n_improved - n_regressed

    if n_compared == 0:
        summary = "No shared metrics between current and baseline — skipped regression detection"
    elif regression_result.regression_detected:
        summary = (
            f"REGRESSION DETECTED: {n_regressed} of {n_compared} metrics "
            f"degraded beyond {tolerance:.1%} tolerance"
        )
    else:
        parts = []
        if n_improved:
            parts.append(f"{n_improved} improved")
        if n_unchanged:
            parts.append(f"{n_unchanged} stable")
        summary = f"All {n_compared} metrics within tolerance ({', '.join(parts)})"

    return ModelValidationResult(
        comparisons=comparisons,
        regression_result=regression_result,
        model_name=current.model_name,
        framework=current.framework,
        summary=summary,
        current_only_metrics=current_only,
        baseline_only_metrics=baseline_only,
    )
