"""Tests for release-facing action metadata."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTION_PATH = REPO_ROOT / "action.yml"


def test_action_uses_moving_minor_image_tag() -> None:
    action = yaml.safe_load(ACTION_PATH.read_text(encoding="utf-8"))
    assert action["runs"]["using"] == "docker"
    assert action["runs"]["image"] == "docker://ghcr.io/ml-ci-labs/ml-ci-action:v0.2"
