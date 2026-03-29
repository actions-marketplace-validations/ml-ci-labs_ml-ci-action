"""Tests for release-facing action metadata."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.reporters.pr_comment import VERSION as REPORT_VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTION_PATH = REPO_ROOT / "action.yml"
README_PATH = REPO_ROOT / "README.md"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _project_version() -> str:
    pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert match, "pyproject.toml is missing a project version"
    return match.group(1)


def test_action_uses_moving_minor_image_tag() -> None:
    action = yaml.safe_load(ACTION_PATH.read_text(encoding="utf-8"))
    assert action["runs"]["using"] == "docker"
    assert action["runs"]["image"] == "docker://ghcr.io/ml-ci-labs/ml-ci-action:v0.3"


def test_release_facing_versions_are_consistent() -> None:
    version = _project_version()
    readme = README_PATH.read_text(encoding="utf-8")
    action_refs = re.findall(r"ml-ci-labs/ml-ci-action@v\d+\.\d+\.\d+", readme)

    assert REPORT_VERSION == version
    assert action_refs
    assert set(action_refs) == {f"ml-ci-labs/ml-ci-action@v{version}"}
