"""Metric loading, validation, and comparison for ML CI."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

import requests

class BaselineFetchError(RuntimeError):
    """Raised when a remote baseline cannot be fetched with actionable context."""


# Default metric direction: True = higher is better, False = lower is better
HIGHER_IS_BETTER_DEFAULTS: dict[str, bool] = {
    "accuracy": True,
    "f1": True,
    "f1_score": True,
    "f1_macro": True,
    "f1_micro": True,
    "f1_weighted": True,
    "precision": True,
    "recall": True,
    "auc": True,
    "auc_roc": True,
    "auroc": True,
    "ap": True,
    "average_precision": True,
    "map": True,
    "ndcg": True,
    "bleu": True,
    "rouge": True,
    "rouge1": True,
    "rouge2": True,
    "rougeL": True,
    "r2": True,
    "r2_score": True,
    "iou": True,
    "miou": True,
    "dice": True,
    "top_k_accuracy": True,
    "hit_rate": True,
    "mrr": True,
    # Lower is better
    "loss": False,
    "train_loss": False,
    "val_loss": False,
    "test_loss": False,
    "mse": False,
    "rmse": False,
    "mae": False,
    "mape": False,
    "error": False,
    "error_rate": False,
    "cer": False,
    "wer": False,
    "perplexity": False,
    "brier_score": False,
    "log_loss": False,
    "cross_entropy": False,
    "hinge_loss": False,
    "false_positive_rate": False,
    "false_negative_rate": False,
}


@dataclass
class MetricsData:
    """Container for model metrics loaded from a JSON file."""

    model_name: str
    framework: str = "unknown"
    timestamp: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricComparison:
    """Result of comparing a single metric between current and baseline."""

    name: str
    current: float
    baseline: float
    delta: float  # current - baseline (raw)
    delta_pct: float  # percentage change
    higher_is_better: bool
    improved: bool  # True if change is in the good direction
    regression: bool  # True if degradation exceeds tolerance


def load_metrics(path: str) -> MetricsData:
    """Load and validate a metrics JSON file from disk.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required fields are missing or metrics are not numeric.
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Metrics file must contain a JSON object, got {type(data).__name__}")

    if "metrics" not in data:
        raise ValueError("Metrics file must contain a 'metrics' key")

    metrics = data["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("'metrics' must be a dictionary of metric_name -> numeric_value")

    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"Metric '{key}' must be numeric, got {type(value).__name__}: {value}")

    return MetricsData(
        model_name=data.get("model_name", "unnamed-model"),
        framework=data.get("framework", "unknown"),
        timestamp=data.get("timestamp", ""),
        metrics=metrics,
        dataset=data.get("dataset", {}),
        hyperparameters=data.get("hyperparameters", {}),
    )


def load_metrics_from_github(
    repo: str,
    path: str,
    ref: str,
    token: str,
) -> MetricsData:
    """Fetch a metrics JSON file from a GitHub repo at a specific ref.

    Uses the GitHub Contents API: GET /repos/{owner}/{repo}/contents/{path}?ref={ref}

    Raises:
        FileNotFoundError: If the file does not exist at the given ref.
        ValueError: If the response cannot be parsed.
        BaselineFetchError: If GitHub rejects the fetch for permissions or size reasons.
        requests.HTTPError: On other API errors.
    """
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    params = {"ref": ref}

    resp = requests.get(url, headers=headers, params=params, timeout=30)

    if resp.status_code == 404:
        raise FileNotFoundError(
            f"Metrics file '{path}' not found in {repo} at ref '{ref}'. "
            "This may be the first PR adding metrics — skipping baseline comparison."
        )

    if resp.status_code == 403:
        try:
            message = resp.json().get("message", "")
        except ValueError:
            message = ""
        raise BaselineFetchError(
            f"GitHub rejected the baseline fetch for '{path}' at ref '{ref}' in {repo}. "
            "This is commonly caused by repository permissions or the GitHub Contents API size limit. "
            "Use a local file path for 'baseline-metrics' instead of fetching from the branch. "
            f"GitHub response: {message or '403 Forbidden'}"
        )

    resp.raise_for_status()

    content_data = resp.json()
    if content_data.get("encoding") != "base64":
        raise ValueError(f"Unexpected encoding: {content_data.get('encoding')}")

    raw = base64.b64decode(content_data["content"]).decode("utf-8")
    data = json.loads(raw)

    # Reuse the same validation as load_metrics
    if "metrics" not in data:
        raise ValueError("Baseline metrics file must contain a 'metrics' key")

    metrics = data["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("'metrics' must be a dictionary")

    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"Baseline metric '{key}' must be numeric, got {type(value).__name__}")

    return MetricsData(
        model_name=data.get("model_name", "unnamed-model"),
        framework=data.get("framework", "unknown"),
        timestamp=data.get("timestamp", ""),
        metrics=metrics,
        dataset=data.get("dataset", {}),
        hyperparameters=data.get("hyperparameters", {}),
    )


def _is_higher_better(metric_name: str) -> bool:
    """Infer whether higher values are better for a given metric name."""
    normalized = metric_name.lower().strip()
    if normalized in HIGHER_IS_BETTER_DEFAULTS:
        return HIGHER_IS_BETTER_DEFAULTS[normalized]
    # Heuristic: if the name contains "loss", "error", or "cost", lower is better
    for keyword in ("loss", "error", "cost", "divergence"):
        if keyword in normalized:
            return False
    # Default: higher is better
    return True


def compare_metrics(
    current: MetricsData,
    baseline: MetricsData,
    tolerance: float = 0.02,
    higher_is_better: dict[str, bool] | None = None,
) -> list[MetricComparison]:
    """Compare each metric between current and baseline models.

    Only compares metrics present in both current and baseline.

    Args:
        current: Current model metrics.
        baseline: Baseline model metrics.
        tolerance: Maximum allowed degradation as a fraction (e.g., 0.02 = 2%).
        higher_is_better: Optional override dict for metric direction.

    Returns:
        List of MetricComparison objects, one per shared metric.
    """
    overrides = higher_is_better or {}
    comparisons: list[MetricComparison] = []

    shared_metrics = set(current.metrics.keys()) & set(baseline.metrics.keys())

    for name in sorted(shared_metrics):
        cur_val = current.metrics[name]
        base_val = baseline.metrics[name]
        delta = cur_val - base_val

        if base_val != 0:
            delta_pct = delta / abs(base_val)
        else:
            delta_pct = 0.0 if delta == 0 else float("inf")

        hib = overrides.get(name, _is_higher_better(name))

        # Did the metric improve?
        if hib:
            improved = delta >= 0
            regression = delta_pct < -tolerance
        else:
            improved = delta <= 0
            regression = delta_pct > tolerance

        comparisons.append(
            MetricComparison(
                name=name,
                current=cur_val,
                baseline=base_val,
                delta=delta,
                delta_pct=delta_pct,
                higher_is_better=hib,
                improved=improved,
                regression=regression,
            )
        )

    return comparisons
