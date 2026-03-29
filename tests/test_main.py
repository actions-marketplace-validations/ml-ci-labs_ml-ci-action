"""Tests for the action entry point."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src import main as main_module


def _write_metrics_file(path: Path) -> None:
    _write_custom_metrics_file(path, framework="pytorch")


def _write_custom_metrics_file(path: Path, framework: str | None) -> None:
    payload = {
        "model_name": "demo-model",
        "metrics": {"accuracy": 0.95, "loss": 0.12},
    }
    if framework is not None:
        payload["framework"] = framework
    path.write_text(json.dumps(payload))


def _write_metrics_with_observations(path: Path, accuracy_obs: list[float], loss_obs: list[float]) -> None:
    path.write_text(
        json.dumps(
            {
                "model_name": "demo-model",
                "framework": "pytorch",
                "metrics": {
                    "accuracy": sum(accuracy_obs) / len(accuracy_obs),
                    "loss": sum(loss_obs) / len(loss_obs),
                },
                "observations": {
                    "accuracy": accuracy_obs,
                    "loss": loss_obs,
                },
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


def test_framework_input_fills_missing_metrics_framework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    metrics_path = tmp_path / "metrics.json"
    _write_custom_metrics_file(metrics_path, framework=None)

    output_path = tmp_path / "github_output.txt"
    _set_common_env(monkeypatch, tmp_path, output_path)
    monkeypatch.setenv("INPUT_METRICS_FILE", "metrics.json")
    monkeypatch.setenv("INPUT_FRAMEWORK", "sklearn")
    monkeypatch.setenv("INPUT_COMMENT_ON_PR", "false")

    main_module.main()

    outputs = _parse_outputs(output_path)
    report = json.loads(outputs["report-json"])
    assert report["framework"] == "sklearn"


def test_framework_input_does_not_override_metrics_framework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    metrics_path = tmp_path / "metrics.json"
    _write_custom_metrics_file(metrics_path, framework="pytorch")

    output_path = tmp_path / "github_output.txt"
    _set_common_env(monkeypatch, tmp_path, output_path)
    monkeypatch.setenv("INPUT_METRICS_FILE", "metrics.json")
    monkeypatch.setenv("INPUT_FRAMEWORK", "sklearn")
    monkeypatch.setenv("INPUT_COMMENT_ON_PR", "false")

    main_module.main()

    outputs = _parse_outputs(output_path)
    report = json.loads(outputs["report-json"])
    assert report["framework"] == "pytorch"


def test_repo_policy_file_is_auto_discovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    metrics_path = tmp_path / "metrics.json"
    _write_metrics_file(metrics_path)
    baseline_path = tmp_path / "baseline.json"
    _write_metrics_file(baseline_path)
    (tmp_path / ".ml-ci.yml").write_text(
        "\n".join(
            [
                "version: 1",
                "policy:",
                "  regression_test: threshold",
                "  regression_tolerance: 0.01",
                "  metrics:",
                "    accuracy:",
                "      tolerance: 0.10",
                "      direction: higher",
                "      severity: warn",
            ]
        )
    )

    output_path = tmp_path / "github_output.txt"
    _set_common_env(monkeypatch, tmp_path, output_path)
    monkeypatch.setenv("INPUT_METRICS_FILE", "metrics.json")
    monkeypatch.setenv("INPUT_BASELINE_METRICS", "baseline.json")
    monkeypatch.setenv("INPUT_COMMENT_ON_PR", "false")

    main_module.main()

    outputs = _parse_outputs(output_path)
    assert outputs["validation-passed"] == "true"
    report = json.loads(outputs["report-json"])
    assert report["regression_test"]["method"] == "threshold"
    assert report["comparisons"][0]["severity"] == "warn" or any(
        item["severity"] == "warn" for item in report["comparisons"]
    )


def test_workflow_inputs_override_repo_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    current_path = tmp_path / "current.json"
    _write_metrics_with_observations(
        current_path,
        accuracy_obs=[0.95, 0.96, 0.94, 0.95, 0.96],
        loss_obs=[0.15, 0.14, 0.16, 0.15, 0.14],
    )
    baseline_path = tmp_path / "baseline.json"
    _write_metrics_with_observations(
        baseline_path,
        accuracy_obs=[0.93, 0.94, 0.92, 0.93, 0.94],
        loss_obs=[0.18, 0.17, 0.19, 0.18, 0.17],
    )
    (tmp_path / ".ml-ci.yml").write_text(
        "version: 1\npolicy:\n  regression_test: threshold\n  regression_tolerance: 0.01\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "github_output.txt"
    _set_common_env(monkeypatch, tmp_path, output_path)
    monkeypatch.setenv("INPUT_METRICS_FILE", "current.json")
    monkeypatch.setenv("INPUT_BASELINE_METRICS", "baseline.json")
    monkeypatch.setenv("INPUT_REGRESSION_TEST", "wilcoxon")
    monkeypatch.setenv("INPUT_COMMENT_ON_PR", "false")

    main_module.main()

    outputs = _parse_outputs(output_path)
    report = json.loads(outputs["report-json"])
    assert report["regression_test"]["method"] == "wilcoxon"


def test_invalid_repo_policy_fails_with_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    metrics_path = tmp_path / "metrics.json"
    _write_metrics_file(metrics_path)
    (tmp_path / ".ml-ci.yml").write_text(
        "version: 1\npolicy:\n  metrics:\n    accuracy:\n      severity: info\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "github_output.txt"
    _set_common_env(monkeypatch, tmp_path, output_path)
    monkeypatch.setenv("INPUT_METRICS_FILE", "metrics.json")
    monkeypatch.setenv("INPUT_COMMENT_ON_PR", "false")

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 1
    outputs = _parse_outputs(output_path)
    assert outputs["validation-passed"] == "false"
    assert "severity" in outputs["report-json"]


def test_warn_only_regression_sets_non_blocking_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    current_path = tmp_path / "current.json"
    _write_metrics_file(current_path)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "model_name": "demo-model",
                "framework": "pytorch",
                "metrics": {"accuracy": 0.97, "loss": 0.12},
            }
        )
    )
    (tmp_path / ".ml-ci.yml").write_text(
        "\n".join(
            [
                "version: 1",
                "policy:",
                "  regression_tolerance: 0.01",
                "  metrics:",
                "    accuracy:",
                "      severity: warn",
                "      direction: higher",
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "github_output.txt"
    _set_common_env(monkeypatch, tmp_path, output_path)
    monkeypatch.setenv("INPUT_METRICS_FILE", "current.json")
    monkeypatch.setenv("INPUT_BASELINE_METRICS", "baseline.json")
    monkeypatch.setenv("INPUT_COMMENT_ON_PR", "false")

    main_module.main()

    outputs = _parse_outputs(output_path)
    assert outputs["validation-passed"] == "true"
    assert outputs["regression-detected"] == "true"
    report = json.loads(outputs["report-json"])
    assert report["blocking_regression_detected"] is False


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


def test_wilcoxon_method_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    current_path = tmp_path / "current.json"
    _write_metrics_with_observations(
        current_path,
        accuracy_obs=[0.95, 0.96, 0.94, 0.95, 0.96],
        loss_obs=[0.15, 0.14, 0.16, 0.15, 0.14],
    )
    baseline_path = tmp_path / "baseline.json"
    _write_metrics_with_observations(
        baseline_path,
        accuracy_obs=[0.93, 0.94, 0.92, 0.93, 0.94],
        loss_obs=[0.18, 0.17, 0.19, 0.18, 0.17],
    )

    output_path = tmp_path / "github_output.txt"
    _set_common_env(monkeypatch, tmp_path, output_path)
    monkeypatch.setenv("INPUT_METRICS_FILE", "current.json")
    monkeypatch.setenv("INPUT_BASELINE_METRICS", "baseline.json")
    monkeypatch.setenv("INPUT_REGRESSION_TEST", "wilcoxon")
    monkeypatch.setenv("INPUT_COMMENT_ON_PR", "false")

    main_module.main()

    outputs = _parse_outputs(output_path)
    assert outputs["validation-passed"] == "true"
    assert outputs["regression-detected"] == "false"
    report = json.loads(outputs["report-json"])
    assert report["regression_test"]["method"] == "wilcoxon"


def test_bootstrap_method_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    current_path = tmp_path / "current.json"
    _write_metrics_with_observations(
        current_path,
        accuracy_obs=[0.95, 0.96, 0.94, 0.95, 0.96],
        loss_obs=[0.15, 0.14, 0.16, 0.15, 0.14],
    )
    baseline_path = tmp_path / "baseline.json"
    _write_metrics_with_observations(
        baseline_path,
        accuracy_obs=[0.93, 0.94, 0.92, 0.93, 0.94],
        loss_obs=[0.18, 0.17, 0.19, 0.18, 0.17],
    )

    output_path = tmp_path / "github_output.txt"
    _set_common_env(monkeypatch, tmp_path, output_path)
    monkeypatch.setenv("INPUT_METRICS_FILE", "current.json")
    monkeypatch.setenv("INPUT_BASELINE_METRICS", "baseline.json")
    monkeypatch.setenv("INPUT_REGRESSION_TEST", "bootstrap")
    monkeypatch.setenv("INPUT_COMMENT_ON_PR", "false")

    main_module.main()

    outputs = _parse_outputs(output_path)
    assert outputs["validation-passed"] == "true"
    assert outputs["regression-detected"] == "false"
    report = json.loads(outputs["report-json"])
    assert report["regression_test"]["method"] == "bootstrap"


def test_statistical_method_without_observations_fails_with_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    current_path = tmp_path / "current.json"
    _write_metrics_file(current_path)
    baseline_path = tmp_path / "baseline.json"
    _write_metrics_file(baseline_path)

    output_path = tmp_path / "github_output.txt"
    _set_common_env(monkeypatch, tmp_path, output_path)
    monkeypatch.setenv("INPUT_METRICS_FILE", "current.json")
    monkeypatch.setenv("INPUT_BASELINE_METRICS", "baseline.json")
    monkeypatch.setenv("INPUT_REGRESSION_TEST", "wilcoxon")
    monkeypatch.setenv("INPUT_COMMENT_ON_PR", "false")

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 1
    outputs = _parse_outputs(output_path)
    assert outputs["validation-passed"] == "false"
