"""Statistical tests for model regression detection."""

from __future__ import annotations

import warnings
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
    observations: dict[str, tuple[list[float], list[float]]],
    higher_is_better: dict[str, bool],
    alpha: float = 0.05,
) -> RegressionResult:
    """Wilcoxon signed-rank test for paired metric observations.

    Runs a separate test per metric. A regression is detected if ANY metric
    shows a statistically significant degradation.

    Args:
        observations: Dict mapping metric name to (current_values, baseline_values).
        higher_is_better: Dict mapping metric name to direction.
        alpha: Significance level (default 0.05).

    Returns:
        RegressionResult with per-metric p-values and regression flags.
    """
    import numpy as np
    from scipy.stats import wilcoxon as scipy_wilcoxon

    metric_results: dict[str, dict[str, Any]] = {}
    regressed_metrics: list[dict[str, Any]] = []

    for name, (cur_vals, base_vals) in sorted(observations.items()):
        cur = np.array(cur_vals, dtype=float)
        base = np.array(base_vals, dtype=float)
        diffs = cur - base
        median_diff = float(np.median(diffs))
        n_obs = len(diffs)
        hib = higher_is_better.get(name, True)

        # If all differences are zero, no change detected
        if np.all(diffs == 0):
            metric_results[name] = {
                "p_value": 1.0,
                "statistic": 0.0,
                "median_diff": 0.0,
                "n_observations": n_obs,
                "significant": False,
                "regressed": False,
            }
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p_value = scipy_wilcoxon(diffs, alternative="two-sided")

        significant = bool(p_value < alpha)
        # Regression = significant AND change is in the bad direction
        if hib:
            regressed = significant and median_diff < 0
        else:
            regressed = significant and median_diff > 0

        result_entry = {
            "p_value": float(p_value),
            "statistic": float(stat),
            "median_diff": median_diff,
            "n_observations": n_obs,
            "significant": significant,
            "regressed": regressed,
        }
        metric_results[name] = result_entry

        if regressed:
            regressed_metrics.append({"metric": name, **result_entry})

    return RegressionResult(
        method="wilcoxon",
        regression_detected=len(regressed_metrics) > 0,
        details={
            "alpha": alpha,
            "total_metrics": len(observations),
            "regressed_count": len(regressed_metrics),
            "regressed_metrics": regressed_metrics,
            "metric_results": metric_results,
        },
    )


def bootstrap_ci(
    observations: dict[str, tuple[list[float], list[float]]],
    higher_is_better: dict[str, bool],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> RegressionResult:
    """Bootstrap confidence interval for mean metric difference.

    Computes a CI for the mean paired difference per metric. A regression is
    detected if the entire CI is on the "bad" side of zero.

    Args:
        observations: Dict mapping metric name to (current_values, baseline_values).
        higher_is_better: Dict mapping metric name to direction.
        n_bootstrap: Number of bootstrap resamples (default 10000).
        confidence: Confidence level (default 0.95).
        seed: Random seed for reproducibility (default None).

    Returns:
        RegressionResult with per-metric CI bounds and regression flags.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    metric_results: dict[str, dict[str, Any]] = {}
    regressed_metrics: list[dict[str, Any]] = []

    lower_pct = (1 - confidence) / 2 * 100
    upper_pct = (1 + confidence) / 2 * 100

    for name, (cur_vals, base_vals) in sorted(observations.items()):
        cur = np.array(cur_vals, dtype=float)
        base = np.array(base_vals, dtype=float)
        diffs = cur - base
        mean_diff = float(np.mean(diffs))
        n_obs = len(diffs)
        hib = higher_is_better.get(name, True)

        # Bootstrap resample
        boot_indices = rng.integers(0, n_obs, size=(n_bootstrap, n_obs))
        boot_means = np.mean(diffs[boot_indices], axis=1)
        ci_lower = float(np.percentile(boot_means, lower_pct))
        ci_upper = float(np.percentile(boot_means, upper_pct))

        # Regression = entire CI on the bad side of zero
        if hib:
            regressed = ci_upper < 0
        else:
            regressed = ci_lower > 0

        result_entry = {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mean_diff": mean_diff,
            "n_observations": n_obs,
            "n_bootstrap": n_bootstrap,
            "regressed": regressed,
        }
        metric_results[name] = result_entry

        if regressed:
            regressed_metrics.append({"metric": name, **result_entry})

    return RegressionResult(
        method="bootstrap",
        regression_detected=len(regressed_metrics) > 0,
        details={
            "confidence": confidence,
            "n_bootstrap": n_bootstrap,
            "total_metrics": len(observations),
            "regressed_count": len(regressed_metrics),
            "regressed_metrics": regressed_metrics,
            "metric_results": metric_results,
        },
    )
