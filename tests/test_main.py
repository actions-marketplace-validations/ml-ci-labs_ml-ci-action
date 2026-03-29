"""Tests for the action entry point."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src import main as main_module


def _write_metrics_file(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "model_name": "demo-model",
                "framework": "pytorch",
                "metrics": {"accuracy": 0.95, "loss": 0.12},
            }
        )
    )


def _set_common_env(monkeypatch: pytest.MonkeyPatch, workspace: Path, output_path: Path) -> None:
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)


def _parse_outputs(path: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    lines = path.read_text().splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        if "<<" in line:
            name, delimiter = line.split("<<", 1)
            i += 1
            value_lines: list[str] = []
            while i < len(lines) and lines[i] != delimiter:
                value_lines.append(lines[i])
                i += 1
            outputs[name] = "\n".join(value_lines)
        elif "=" in line:
            name, value = line.split("=", 1)
            outputs[name] = value
        i += 1

    return outputs


def test_data_validation_failure_sets_outputs_before_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    metrics_path = tmp_path / "metrics.json"
    _write_metrics_file(metrics_path)

    data_path = tmp_path / "bad_data.csv"
    data_path.write_text("feature_a,feature_b\n,\n,\n,\n")

    output_path = tmp_path / "github_output.txt"
    _set_common_env(monkeypatch, tmp_path, output_path)
    monkeypatch.setenv("INPUT_METRICS_FILE", "metrics.json")
    monkeypatch.setenv("INPUT_DATA_PATH", "bad_data.csv")
    monkeypatch.setenv("INPUT_FAIL_ON_REGRESSION", "true")
    monkeypatch.setenv("INPUT_COMMENT_ON_PR", "false")

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 1
    outputs = _parse_outputs(output_path)
    assert outputs["validation-passed"] == "false"
    assert outputs["regression-detected"] == "false"
    assert outputs["model-card-path"] == ""
    assert '"validation_passed": false' in outputs["report-json"]


def test_get_input_accepts_hyphenated_docker_action_env_names(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INPUT_METRICS-FILE", "metrics.json")
    monkeypatch.delenv("INPUT_METRICS_FILE", raising=False)

    assert main_module.get_input("METRICS-FILE") == "metrics.json"


def test_model_card_generation_failure_degrades_gracefully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    metrics_path = tmp_path / "metrics.json"
    _write_metrics_file(metrics_path)

    output_path = tmp_path / "github_output.txt"
    _set_common_env(monkeypatch, tmp_path, output_path)
    monkeypatch.setenv("INPUT_METRICS_FILE", "metrics.json")
    monkeypatch.setenv("INPUT_MODEL_CARD", "true")
    monkeypatch.setenv("INPUT_COMMENT_ON_PR", "false")

    import src.reporters.model_card as model_card_module

    def _raise_model_card_error(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(model_card_module, "generate_model_card", _raise_model_card_error)

    main_module.main()

    outputs = _parse_outputs(output_path)
    assert outputs["validation-passed"] == "true"
    assert outputs["model-card-path"] == ""


def test_pr_comment_skipped_outside_pr_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    metrics_path = tmp_path / "metrics.json"
    _write_metrics_file(metrics_path)

    output_path = tmp_path / "github_output.txt"
    _set_common_env(monkeypatch, tmp_path, output_path)
    monkeypatch.setenv("INPUT_METRICS_FILE", "metrics.json")
    monkeypatch.setenv("INPUT_COMMENT_ON_PR", "true")
    monkeypatch.setenv("INPUT_GITHUB_TOKEN", "token123")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    import src.reporters.pr_comment as pr_comment_module

    mocked_comment = MagicMock()
    monkeypatch.setattr(pr_comment_module, "post_or_update_comment", mocked_comment)

    main_module.main()

    mocked_comment.assert_not_called()


def test_missing_remote_baseline_gracefully_falls_back_to_current_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    metrics_path = tmp_path / "metrics.json"
    _write_metrics_file(metrics_path)

    output_path = tmp_path / "github_output.txt"
    _set_common_env(monkeypatch, tmp_path, output_path)
    monkeypatch.setenv("INPUT_METRICS_FILE", "metrics.json")
    monkeypatch.setenv("INPUT_BASELINE_METRICS", "main")
    monkeypatch.setenv("INPUT_COMMENT_ON_PR", "false")
    monkeypatch.setenv("INPUT_GITHUB_TOKEN", "token123")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    import src.utils.metrics as metrics_module

    def _raise_missing(*_args, **_kwargs):
        raise FileNotFoundError("missing on main")

    monkeypatch.setattr(metrics_module, "load_metrics_from_github", _raise_missing)

    main_module.main()

    outputs = _parse_outputs(output_path)
    assert outputs["validation-passed"] == "true"
    assert outputs["regression-detected"] == "false"
    assert '"current_metrics"' in outputs["report-json"]


def test_permission_denied_remote_baseline_gracefully_falls_back_to_current_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    metrics_path = tmp_path / "metrics.json"
    _write_metrics_file(metrics_path)

    output_path = tmp_path / "github_output.txt"
    _set_common_env(monkeypatch, tmp_path, output_path)
    monkeypatch.setenv("INPUT_METRICS_FILE", "metrics.json")
    monkeypatch.setenv("INPUT_BASELINE_METRICS", "main")
    monkeypatch.setenv("INPUT_COMMENT_ON_PR", "false")
    monkeypatch.setenv("INPUT_GITHUB_TOKEN", "token123")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    import src.utils.metrics as metrics_module

    def _raise_forbidden(*_args, **_kwargs):
        raise metrics_module.BaselineFetchError("Use a local file path for 'baseline-metrics'")

    monkeypatch.setattr(metrics_module, "load_metrics_from_github", _raise_forbidden)

    main_module.main()

    outputs = _parse_outputs(output_path)
    assert outputs["validation-passed"] == "true"
    assert outputs["regression-detected"] == "false"
