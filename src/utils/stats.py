"""Statistical tests for model regression detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.utils.metrics import MetricComparison


@dataclass
class RegressionResult:
    """Result of a regression detection test."""

    method: str  # "threshold", "wilcoxon", "bootstrap"
    regression_detected: bool
    details: dict[str, Any] = field(default_factory=dict)


def threshold_test(
    comparisons: list[MetricComparison],
    tolerance: float = 0.02,
) -> RegressionResult:
    """Simple threshold-based regression detection.

    A regression is detected if ANY metric degrades by more than the
    specified tolerance (as a fraction of the baseline value).

    Args:
        comparisons: List of MetricComparison from compare_metrics().
        tolerance: Maximum allowed degradation fraction (default 2%).

    Returns:
        RegressionResult with details about which metrics regressed.
    """
    regressed_metrics: list[dict[str, Any]] = []

    for comp in comparisons:
        if comp.regression:
            regressed_metrics.append({
                "metric": comp.name,
                "current": comp.current,
                "baseline": comp.baseline,
                "delta": comp.delta,
                "delta_pct": comp.delta_pct,
                "higher_is_better": comp.higher_is_better,
            })

    return RegressionResult(
        method="threshold",
        regression_detected=len(regressed_metrics) > 0,
        details={
            "tolerance": tolerance,
            "total_metrics": len(comparisons),
            "regressed_count": len(regressed_metrics),
            "regressed_metrics": regressed_metrics,
        },
    )


def wilcoxon_test(
    current_values: list[float],
    baseline_values: list[float],
    alpha: float = 0.05,
) -> RegressionResult:
    """Wilcoxon signed-rank test for paired metric observations.

    Requires paired observations from cross-validation or repeated evaluation.
    Available in v0.2.

    Raises:
        NotImplementedError: This method is not yet implemented.
    """
    raise NotImplementedError(
        "Wilcoxon signed-rank test will be available in v0.2. "
        "Use 'threshold' regression test for now."
    )


def bootstrap_ci(
    current_values: list[float],
    baseline_values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> RegressionResult:
    """Bootstrap confidence interval for mean metric difference.

    Available in v0.2.

    Raises:
        NotImplementedError: This method is not yet implemented.
    """
    raise NotImplementedError(
        "Bootstrap confidence intervals will be available in v0.2. "
        "Use 'threshold' regression test for now."
    )
