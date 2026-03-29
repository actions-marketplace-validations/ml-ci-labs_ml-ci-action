"""Build the versioned payload uploaded to the future ML-CI App."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from src.utils.metrics import MetricsData
from src.version import VERSION


PAYLOAD_SCHEMA_VERSION = 1


def build_run_payload(
    *,
    current: MetricsData,
    report_data: dict[str, Any],
    model_card_path: str | None = None,
) -> dict[str, Any]:
    """Assemble the versioned payload contract for upload."""
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "source": {
            "provider": "github-actions",
            "action_version": VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "repository": _repository_metadata(),
        "git": _git_metadata(),
        "model": {
            "name": current.model_name,
            "framework": current.framework,
            "timestamp": current.timestamp or None,
        },
        "metrics": {
            "current": current.metrics,
            "shared": report_data.get("shared_metrics", []),
            "current_only": report_data.get("current_only_metrics", []),
            "baseline_only": report_data.get("baseline_only_metrics", []),
            "comparisons": report_data.get("comparisons", []),
            "regression_test": report_data.get("regression_test"),
        },
        "validation": {
            "validation_passed": report_data.get("validation_passed", False),
            "regression_detected": report_data.get("regression_detected", False),
            "blocking_regression_detected": report_data.get(
                "blocking_regression_detected",
                False,
            ),
            "baseline_source": report_data.get("baseline_source"),
            "data_policy": report_data.get("data_policy"),
            "data_validation": report_data.get("data_validation"),
        },
        "model_card": {
            "generated": bool(model_card_path),
            "path": model_card_path or None,
            "content": _read_optional_text(model_card_path),
        },
        "lineage": {
            "dataset": current.dataset,
            "training_data_hash": current.lineage.get("training_data_hash"),
            "model_artifact_hash": current.lineage.get("model_artifact_hash"),
            "hyperparameters": current.hyperparameters,
            "environment": current.lineage.get("environment", {}),
        },
    }


def _repository_metadata() -> dict[str, str | None]:
    full_name = os.environ.get("GITHUB_REPOSITORY", "")
    owner = None
    name = None
    if "/" in full_name:
        owner, name = full_name.split("/", 1)
    return {
        "full_name": full_name or None,
        "owner": owner,
        "name": name,
    }


def _git_metadata() -> dict[str, Any]:
    event = _read_event_payload()
    return {
        "sha": os.environ.get("GITHUB_SHA") or None,
        "ref": os.environ.get("GITHUB_REF") or None,
        "head_ref": os.environ.get("GITHUB_HEAD_REF") or None,
        "base_ref": os.environ.get("GITHUB_BASE_REF") or None,
        "event_name": os.environ.get("GITHUB_EVENT_NAME") or None,
        "pull_request_number": _pull_request_number(event),
        "run_id": os.environ.get("GITHUB_RUN_ID") or None,
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT") or None,
        "run_number": os.environ.get("GITHUB_RUN_NUMBER") or None,
        "workflow": os.environ.get("GITHUB_WORKFLOW") or None,
        "job": os.environ.get("GITHUB_JOB") or None,
        "actor": os.environ.get("GITHUB_ACTOR") or None,
    }


def _read_event_payload() -> dict[str, Any]:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return {}

    try:
        return json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _pull_request_number(event: dict[str, Any]) -> int | None:
    pull_request = event.get("pull_request")
    if isinstance(pull_request, dict):
        number = pull_request.get("number")
        return number if isinstance(number, int) else None

    issue = event.get("issue")
    if isinstance(issue, dict) and "pull_request" in issue:
        number = issue.get("number")
        return number if isinstance(number, int) else None

    return None


def _read_optional_text(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
