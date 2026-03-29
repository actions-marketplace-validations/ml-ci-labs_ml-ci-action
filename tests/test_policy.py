"""Tests for repo policy config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.policy import (
    PolicyConfigError,
    discover_policy_file,
    load_policy_config,
    resolve_policy,
    WorkflowPolicyOverrides,
)


def test_discover_policy_file_finds_repo_root_config(tmp_path: Path) -> None:
    config_path = tmp_path / ".ml-ci.yml"
    config_path.write_text("version: 1\npolicy: {}\n", encoding="utf-8")

    assert discover_policy_file(tmp_path) == config_path


def test_load_minimal_policy_config(tmp_path: Path) -> None:
    config_path = tmp_path / ".ml-ci.yml"
    config_path.write_text(
        "version: 1\npolicy:\n  regression_test: threshold\n  regression_tolerance: 0.03\n",
        encoding="utf-8",
    )

    config = load_policy_config(config_path)

    assert config.regression_test == "threshold"
    assert config.regression_tolerance == 0.03
    assert config.metric_policies == {}


def test_load_advanced_policy_config(tmp_path: Path) -> None:
    config_path = tmp_path / ".ml-ci.yml"
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "policy:",
                "  regression_test: wilcoxon",
                "  regression_tolerance: 0.05",
                "  higher_is_better:",
                "    accuracy: true",
                "  metrics:",
                "    accuracy:",
                "      tolerance: 0.01",
                "      direction: higher",
                "      severity: fail",
                "    loss:",
                "      tolerance: 0.02",
                "      direction: lower",
                "      severity: warn",
            ]
        ),
        encoding="utf-8",
    )

    config = load_policy_config(config_path)
    resolved = resolve_policy(config, WorkflowPolicyOverrides())

    assert resolved.regression_test == "wilcoxon"
    assert resolved.regression_tolerance == 0.05
    assert resolved.higher_is_better["accuracy"] is True
    assert resolved.higher_is_better["loss"] is False
    assert resolved.metric_tolerances["accuracy"] == 0.01
    assert resolved.metric_severities["loss"] == "warn"


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("version: 2\npolicy: {}\n", "must be the integer 1"),
        (
            "version: 1\npolicy:\n  metrics:\n    accuracy:\n      severity: info\n",
            "severity must be one of",
        ),
        (
            "version: 1\npolicy:\n  metrics:\n    accuracy:\n      direction: sideways\n",
            "direction must be one of",
        ),
        ("version: 1\npolicy:\n  regression_tolerance: nope\n", "expected a non-negative number"),
        ("version: 1\nfoo: bar\n", "unknown key"),
    ],
)
def test_invalid_policy_config_is_actionable(
    tmp_path: Path,
    content: str,
    match: str,
) -> None:
    config_path = tmp_path / ".ml-ci.yml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(PolicyConfigError, match=match):
        load_policy_config(config_path)


def test_workflow_overrides_take_precedence(tmp_path: Path) -> None:
    config_path = tmp_path / ".ml-ci.yml"
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "policy:",
                "  regression_test: wilcoxon",
                "  regression_tolerance: 0.05",
                "  higher_is_better:",
                "    custom_metric: true",
                "  metrics:",
                "    custom_metric:",
                "      tolerance: 0.01",
                "      direction: higher",
                "      severity: warn",
            ]
        ),
        encoding="utf-8",
    )

    config = load_policy_config(config_path)
    resolved = resolve_policy(
        config,
        WorkflowPolicyOverrides(
            regression_test="threshold",
            regression_tolerance=0.02,
            higher_is_better={"custom_metric": False},
        ),
    )

    assert resolved.regression_test == "threshold"
    assert resolved.regression_tolerance == 0.02
    assert resolved.higher_is_better["custom_metric"] is False
    assert resolved.metric_tolerances["custom_metric"] == 0.01
    assert resolved.metric_severities["custom_metric"] == "warn"
