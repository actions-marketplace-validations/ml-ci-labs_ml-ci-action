"""Tests for README-linked example workflows."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
EXAMPLE_WORKFLOWS = {
    "first_run": REPO_ROOT / "examples/pytorch-classification/.github/workflows/ml-ci.yml",
    "branch_baseline": REPO_ROOT / "examples/sklearn-regression/.github/workflows/ml-ci.yml",
    "cross_validation": REPO_ROOT / "examples/cross-validation/.github/workflows/ml-ci.yml",
    "config_minimal": REPO_ROOT / "examples/config-minimal/.github/workflows/ml-ci.yml",
    "config_advanced": REPO_ROOT / "examples/config-advanced/.github/workflows/ml-ci.yml",
}


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict), f"{path} did not parse to a mapping"
    return data


def _expected_action_ref() -> str:
    pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert match, "pyproject.toml is missing a project version"
    version = match.group(1)
    return f"ml-ci-labs/ml-ci-action@v{version}"


def test_example_workflows_exist() -> None:
    for name, path in EXAMPLE_WORKFLOWS.items():
        assert path.exists(), f"Missing example workflow for {name}: {path}"


def test_readme_links_only_existing_example_workflows() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    for path in EXAMPLE_WORKFLOWS.values():
        relative = f"./{path.relative_to(REPO_ROOT).as_posix()}"
        assert relative in readme, f"README is missing example link: {relative}"


def test_first_run_example_is_baseline_free() -> None:
    workflow = _load_yaml(EXAMPLE_WORKFLOWS["first_run"])
    validate_job = workflow["jobs"]["validate"]
    step = validate_job["steps"][-1]
    assert step["uses"] == _expected_action_ref()
    assert "baseline-metrics" not in step["with"]
    assert step["with"]["comment-on-pr"] == "true"


def test_branch_baseline_example_uses_default_branch_comparison() -> None:
    workflow = _load_yaml(EXAMPLE_WORKFLOWS["branch_baseline"])
    step = workflow["jobs"]["validate"]["steps"][-1]
    assert step["uses"] == _expected_action_ref()
    assert step["with"]["baseline-metrics"] == "main"
    assert step["with"]["comment-on-pr"] == "true"


def test_cross_validation_example_uses_statistical_regression() -> None:
    workflow = _load_yaml(EXAMPLE_WORKFLOWS["cross_validation"])
    step = workflow["jobs"]["validate"]["steps"][-1]
    assert step["uses"] == _expected_action_ref()
    assert step["with"]["baseline-metrics"] == "main"
    assert step["with"]["regression-test"] == "wilcoxon"
    assert step["with"]["alpha"] == "0.05"


def test_config_minimal_example_relies_on_repo_policy_file() -> None:
    workflow = _load_yaml(EXAMPLE_WORKFLOWS["config_minimal"])
    step = workflow["jobs"]["validate"]["steps"][-1]
    assert step["uses"] == _expected_action_ref()
    assert "regression-test" not in step["with"]
    assert "regression-tolerance" not in step["with"]
    assert (EXAMPLE_WORKFLOWS["config_minimal"].parents[2] / ".ml-ci.yml").exists()


def test_config_advanced_example_relies_on_repo_policy_file() -> None:
    workflow = _load_yaml(EXAMPLE_WORKFLOWS["config_advanced"])
    step = workflow["jobs"]["validate"]["steps"][-1]
    assert step["uses"] == _expected_action_ref()
    assert step["with"]["baseline-metrics"] == "main"
    assert "higher-is-better" not in step["with"]
    assert (EXAMPLE_WORKFLOWS["config_advanced"].parents[2] / ".ml-ci.yml").exists()
