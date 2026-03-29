"""Policy config discovery, validation, and resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


DEFAULT_POLICY_FILE = ".ml-ci.yml"
SUPPORTED_REGRESSION_TESTS = {"threshold", "wilcoxon", "bootstrap"}
SUPPORTED_SEVERITIES = {"fail", "warn"}
SUPPORTED_DIRECTIONS = {"higher", "lower"}


class PolicyConfigError(ValueError):
    """Raised when the repo policy file is invalid."""


@dataclass(frozen=True)
class MetricPolicy:
    """Per-metric policy overrides."""

    tolerance: float | None = None
    higher_is_better: bool | None = None
    severity: str = "fail"


@dataclass(frozen=True)
class DataPolicy:
    """Data validation policy defaults."""

    missing_threshold: float = 0.2
    missing_thresholds: dict[str, float] = field(default_factory=dict)
    label_column: str | None = None
    include_columns: list[str] = field(default_factory=list)
    exclude_columns: list[str] = field(default_factory=list)


@dataclass
class PolicyConfig:
    """Validated policy config loaded from disk."""

    path: str
    regression_test: str | None = None
    regression_tolerance: float | None = None
    higher_is_better: dict[str, bool] = field(default_factory=dict)
    metric_policies: dict[str, MetricPolicy] = field(default_factory=dict)
    data_policy: DataPolicy = field(default_factory=DataPolicy)


@dataclass(frozen=True)
class WorkflowPolicyOverrides:
    """Policy values explicitly set by workflow inputs."""

    regression_test: str | None = None
    regression_tolerance: float | None = None
    higher_is_better: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedPolicy:
    """Effective policy used during validation."""

    config_path: str | None
    regression_test: str
    regression_tolerance: float
    higher_is_better: dict[str, bool]
    metric_tolerances: dict[str, float]
    metric_severities: dict[str, str]
    data_policy: DataPolicy


def discover_policy_file(workspace: str | Path) -> Path | None:
    """Return the default repo policy file if it exists."""
    policy_path = Path(workspace) / DEFAULT_POLICY_FILE
    return policy_path if policy_path.is_file() else None


def load_policy_config(path: str | Path) -> PolicyConfig:
    """Load and validate a repo policy config."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyConfigError(f"Invalid config at {config_path}: YAML parse error: {exc}") from exc

    if raw is None:
        raise PolicyConfigError(f"Invalid config at {config_path}: file is empty")
    if not isinstance(raw, dict):
        raise PolicyConfigError(
            f"Invalid config at {config_path}: top level must be a mapping, got {type(raw).__name__}"
        )

    _reject_unknown_keys(
        scope=f"Invalid config at {config_path}",
        raw=raw,
        allowed={"version", "policy"},
    )

    version = raw.get("version")
    if version != 1:
        raise PolicyConfigError(
            f"Invalid config at {config_path}: 'version' must be the integer 1, got {version!r}"
        )

    policy_raw = raw.get("policy")
    if policy_raw is None:
        policy_raw = {}
    if not isinstance(policy_raw, dict):
        raise PolicyConfigError(
            f"Invalid config at {config_path}: 'policy' must be a mapping, got {type(policy_raw).__name__}"
        )

    _reject_unknown_keys(
        scope=f"Invalid config at {config_path} policy",
        raw=policy_raw,
        allowed={"regression_test", "regression_tolerance", "higher_is_better", "metrics", "data"},
    )

    regression_test = policy_raw.get("regression_test")
    if regression_test is not None:
        if not isinstance(regression_test, str) or regression_test not in SUPPORTED_REGRESSION_TESTS:
            raise PolicyConfigError(
                f"Invalid config at {config_path}: policy.regression_test must be one of "
                f"{sorted(SUPPORTED_REGRESSION_TESTS)}, got {regression_test!r}"
            )

    regression_tolerance = _validate_optional_tolerance(
        value=policy_raw.get("regression_tolerance"),
        label=f"{config_path}: policy.regression_tolerance",
    )

    higher_is_better = _validate_higher_is_better_map(
        value=policy_raw.get("higher_is_better"),
        label=f"{config_path}: policy.higher_is_better",
    )

    metric_policies = _validate_metric_policies(
        value=policy_raw.get("metrics"),
        config_path=config_path,
    )
    data_policy = _validate_data_policy(
        value=policy_raw.get("data"),
        config_path=config_path,
    )

    return PolicyConfig(
        path=str(config_path),
        regression_test=regression_test,
        regression_tolerance=regression_tolerance,
        higher_is_better=higher_is_better,
        metric_policies=metric_policies,
        data_policy=data_policy,
    )


def resolve_policy(
    config: PolicyConfig | None,
    overrides: WorkflowPolicyOverrides,
) -> ResolvedPolicy:
    """Merge repo config defaults with explicit workflow overrides."""
    regression_test = overrides.regression_test
    if regression_test is None:
        regression_test = config.regression_test if config and config.regression_test else "threshold"

    regression_tolerance = overrides.regression_tolerance
    if regression_tolerance is None:
        regression_tolerance = (
            config.regression_tolerance
            if config and config.regression_tolerance is not None
            else 0.02
        )

    direction_overrides: dict[str, bool] = {}
    metric_tolerances: dict[str, float] = {}
    metric_severities: dict[str, str] = {}

    if config:
        direction_overrides.update(config.higher_is_better)
        for name, metric_policy in config.metric_policies.items():
            if metric_policy.higher_is_better is not None:
                direction_overrides[name] = metric_policy.higher_is_better
            if metric_policy.tolerance is not None:
                metric_tolerances[name] = metric_policy.tolerance
            metric_severities[name] = metric_policy.severity

    direction_overrides.update(overrides.higher_is_better)

    return ResolvedPolicy(
        config_path=config.path if config else None,
        regression_test=regression_test,
        regression_tolerance=regression_tolerance,
        higher_is_better=direction_overrides,
        metric_tolerances=metric_tolerances,
        metric_severities=metric_severities,
        data_policy=config.data_policy if config else DataPolicy(),
    )


def _reject_unknown_keys(scope: str, raw: dict[str, object], allowed: set[str]) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise PolicyConfigError(
            f"{scope}: unknown key(s) {unknown}. Allowed keys: {sorted(allowed)}"
        )


def _validate_optional_tolerance(value: object, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise PolicyConfigError(f"Invalid config at {label}: expected a non-negative number, got {value!r}")
    tolerance = float(value)
    if tolerance < 0:
        raise PolicyConfigError(f"Invalid config at {label}: expected a non-negative number, got {value!r}")
    return tolerance


def _validate_higher_is_better_map(value: object, label: str) -> dict[str, bool]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PolicyConfigError(f"Invalid config at {label}: expected a mapping, got {type(value).__name__}")

    validated: dict[str, bool] = {}
    for metric_name, metric_value in value.items():
        if not isinstance(metric_name, str) or not metric_name:
            raise PolicyConfigError(
                f"Invalid config at {label}: metric names must be non-empty strings, got {metric_name!r}"
            )
        if not isinstance(metric_value, bool):
            raise PolicyConfigError(
                f"Invalid config at {label}.{metric_name}: expected a boolean, got {metric_value!r}"
            )
        validated[metric_name] = metric_value
    return validated


def _validate_metric_policies(value: object, config_path: Path) -> dict[str, MetricPolicy]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PolicyConfigError(
            f"Invalid config at {config_path}: policy.metrics must be a mapping, got {type(value).__name__}"
        )

    validated: dict[str, MetricPolicy] = {}
    for metric_name, metric_policy in value.items():
        if not isinstance(metric_name, str) or not metric_name:
            raise PolicyConfigError(
                f"Invalid config at {config_path}: policy.metrics keys must be non-empty strings, got {metric_name!r}"
            )
        if not isinstance(metric_policy, dict):
            raise PolicyConfigError(
                f"Invalid config at {config_path}: policy.metrics.{metric_name} must be a mapping, "
                f"got {type(metric_policy).__name__}"
            )

        _reject_unknown_keys(
            scope=f"Invalid config at {config_path}: policy.metrics.{metric_name}",
            raw=metric_policy,
            allowed={"tolerance", "direction", "severity"},
        )

        tolerance = _validate_optional_tolerance(
            value=metric_policy.get("tolerance"),
            label=f"{config_path}: policy.metrics.{metric_name}.tolerance",
        )

        direction = metric_policy.get("direction")
        higher_is_better: bool | None = None
        if direction is not None:
            if not isinstance(direction, str) or direction not in SUPPORTED_DIRECTIONS:
                raise PolicyConfigError(
                    f"Invalid config at {config_path}: policy.metrics.{metric_name}.direction must be one of "
                    f"{sorted(SUPPORTED_DIRECTIONS)}, got {direction!r}"
                )
            higher_is_better = direction == "higher"

        severity = metric_policy.get("severity", "fail")
        if not isinstance(severity, str) or severity not in SUPPORTED_SEVERITIES:
            raise PolicyConfigError(
                f"Invalid config at {config_path}: policy.metrics.{metric_name}.severity must be one of "
                f"{sorted(SUPPORTED_SEVERITIES)}, got {severity!r}"
            )

        validated[metric_name] = MetricPolicy(
            tolerance=tolerance,
            higher_is_better=higher_is_better,
            severity=severity,
        )

    return validated


def _validate_data_policy(value: object, config_path: Path) -> DataPolicy:
    if value is None:
        return DataPolicy()
    if not isinstance(value, dict):
        raise PolicyConfigError(
            f"Invalid config at {config_path}: policy.data must be a mapping, got {type(value).__name__}"
        )

    _reject_unknown_keys(
        scope=f"Invalid config at {config_path}: policy.data",
        raw=value,
        allowed={
            "missing_threshold",
            "missing_thresholds",
            "label_column",
            "include_columns",
            "exclude_columns",
        },
    )

    missing_threshold = _validate_optional_tolerance(
        value=value.get("missing_threshold"),
        label=f"{config_path}: policy.data.missing_threshold",
    )
    if missing_threshold is None:
        missing_threshold = 0.2

    missing_thresholds = _validate_missing_threshold_map(
        value=value.get("missing_thresholds"),
        label=f"{config_path}: policy.data.missing_thresholds",
    )
    label_column = _validate_optional_string(
        value=value.get("label_column"),
        label=f"{config_path}: policy.data.label_column",
    )
    include_columns = _validate_string_list(
        value=value.get("include_columns"),
        label=f"{config_path}: policy.data.include_columns",
    )
    exclude_columns = _validate_string_list(
        value=value.get("exclude_columns"),
        label=f"{config_path}: policy.data.exclude_columns",
    )

    overlap = sorted(set(include_columns) & set(exclude_columns))
    if overlap:
        raise PolicyConfigError(
            f"Invalid config at {config_path}: policy.data include/exclude overlap for {overlap!r}"
        )

    return DataPolicy(
        missing_threshold=missing_threshold,
        missing_thresholds=missing_thresholds,
        label_column=label_column,
        include_columns=include_columns,
        exclude_columns=exclude_columns,
    )


def _validate_missing_threshold_map(value: object, label: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PolicyConfigError(f"Invalid config at {label}: expected a mapping, got {type(value).__name__}")

    validated: dict[str, float] = {}
    for column_name, threshold in value.items():
        if not isinstance(column_name, str) or not column_name:
            raise PolicyConfigError(
                f"Invalid config at {label}: column names must be non-empty strings, got {column_name!r}"
            )
        validated[column_name] = _validate_optional_tolerance(
            value=threshold,
            label=f"{label}.{column_name}",
        ) or 0.0
    return validated


def _validate_optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PolicyConfigError(f"Invalid config at {label}: expected a non-empty string, got {value!r}")
    return value.strip()


def _validate_string_list(value: object, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise PolicyConfigError(f"Invalid config at {label}: expected a list, got {type(value).__name__}")

    validated: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise PolicyConfigError(f"Invalid config at {label}: expected non-empty string items, got {item!r}")
        validated.append(item.strip())
    return validated
