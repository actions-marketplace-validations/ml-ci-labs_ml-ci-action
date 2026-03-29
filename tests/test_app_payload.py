"""Tests for the ML-CI App upload payload contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.app_payload import PAYLOAD_SCHEMA_VERSION, build_run_payload
from src.utils.metrics import MetricsData, load_metrics


def _base_report_data() -> dict:
    return {
        "validation_passed": True,
        "regression_detected": False,
        "blocking_regression_detected": False,
        "shared_metrics": ["accuracy", "loss"],
        "current_only_metrics": [],
        "baseline_only_metrics": [],
        "baseline_source": {
            "mode": "remote-legacy",
            "requested_ref": "main",
            "requested_path": "metrics.json",
            "resolved_ref": "main",
            "resolved_path": "metrics.json",
            "available": True,
            "reason": None,
        },
        "comparisons": [
            {
                "name": "accuracy",
                "current": 0.95,
                "baseline": 0.93,
                "delta": 0.02,
                "delta_pct": 0.0215,
                "improved": True,
                "regression": False,
                "severity": "fail",
                "tolerance": 0.02,
            }
        ],
        "regression_test": {
            "method": "threshold",
            "detected": False,
            "blocking_detected": False,
            "details": {"tolerance": 0.02},
        },
        "data_policy": {
            "missing_threshold": 0.2,
            "missing_thresholds": {},
            "label_column": None,
            "include_columns": [],
            "exclude_columns": [],
        },
        "data_validation": None,
    }


def test_build_run_payload_current_only_without_model_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"pull_request": {"number": 42}}), encoding="utf-8")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ml-ci-labs/ml-ci-action")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/run-contract")
    monkeypatch.setenv("GITHUB_BASE_REF", "main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "7")
    monkeypatch.setenv("GITHUB_WORKFLOW", "ML Validation")
    monkeypatch.setenv("GITHUB_JOB", "validate")
    monkeypatch.setenv("GITHUB_ACTOR", "octocat")

    current = MetricsData(
        model_name="fraud-detector-v3",
        framework="pytorch",
        timestamp="2026-03-29T12:34:56Z",
        metrics={"accuracy": 0.95, "loss": 0.15},
        dataset={"name": "transactions-q1-2026"},
        hyperparameters={"epochs": 5},
        lineage={
            "training_data_hash": "sha256:data",
            "model_artifact_hash": "sha256:model",
            "environment": {"python": "3.11.11"},
        },
    )
    report_data = _base_report_data()
    report_data["shared_metrics"] = []
    report_data["current_only_metrics"] = ["accuracy", "loss"]
    report_data["baseline_source"]["mode"] = "none"

    payload = build_run_payload(current=current, report_data=report_data)

    assert payload["schema_version"] == PAYLOAD_SCHEMA_VERSION
    assert payload["repository"]["full_name"] == "ml-ci-labs/ml-ci-action"
    assert payload["git"]["pull_request_number"] == 42
    assert payload["metrics"]["current_only"] == ["accuracy", "loss"]
    assert payload["validation"]["baseline_source"]["mode"] == "none"
    assert payload["model_card"]["generated"] is False
    assert payload["lineage"]["training_data_hash"] == "sha256:data"
    assert payload["lineage"]["environment"]["python"] == "3.11.11"


def test_build_run_payload_includes_model_card_content(tmp_path: Path) -> None:
    card_path = tmp_path / "MODEL_CARD.md"
    card_path.write_text("# Demo card\n", encoding="utf-8")

    current = MetricsData(
        model_name="demo-model",
        framework="pytorch",
        metrics={"accuracy": 0.95},
        dataset={},
        hyperparameters={},
    )

    payload = build_run_payload(
        current=current,
        report_data=_base_report_data(),
        model_card_path=str(card_path),
    )

    assert payload["model_card"]["generated"] is True
    assert payload["model_card"]["path"] == str(card_path)
    assert payload["model_card"]["content"] == "# Demo card\n"


def test_load_metrics_parses_optional_lineage(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_name": "demo-model",
                "metrics": {"accuracy": 0.95},
                "lineage": {
                    "training_data_hash": "sha256:data",
                    "model_artifact_hash": "sha256:model",
                    "environment": {"python": "3.11.11"},
                },
            }
        ),
        encoding="utf-8",
    )

    metrics = load_metrics(str(metrics_path))

    assert metrics.lineage["training_data_hash"] == "sha256:data"
    assert metrics.lineage["model_artifact_hash"] == "sha256:model"
    assert metrics.lineage["environment"]["python"] == "3.11.11"


def test_load_metrics_rejects_invalid_lineage_environment(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_name": "demo-model",
                "metrics": {"accuracy": 0.95},
                "lineage": {"environment": "not-an-object"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="'lineage.environment' must be an object"):
        load_metrics(str(metrics_path))
