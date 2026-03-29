"""Tests for ML-CI App connectivity and upload helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from src.utils.app_client import detect_app_connection, upload_run_payload


def _response(status_code: int, payload: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload or {}
    response.raise_for_status.side_effect = None if status_code < 400 else requests.HTTPError()
    return response


def test_detect_app_connection_succeeds_when_repo_is_accessible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocked_get = MagicMock(
        return_value=_response(
            200,
            {
                "repositories": [
                    {"full_name": "ml-ci-labs/ml-ci-action"},
                    {"full_name": "someone/else"},
                ]
            },
        )
    )
    monkeypatch.setattr("src.utils.app_client.requests.get", mocked_get)

    status = detect_app_connection("ml-ci-labs/ml-ci-action", "token123")

    assert status.connected is True
    assert status.reason is None


def test_detect_app_connection_returns_false_for_inaccessible_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocked_get = MagicMock(
        return_value=_response(
            200,
            {"repositories": [{"full_name": "someone/else"}]},
        )
    )
    monkeypatch.setattr("src.utils.app_client.requests.get", mocked_get)

    status = detect_app_connection("ml-ci-labs/ml-ci-action", "token123")

    assert status.connected is False
    assert status.reason == "repo_not_accessible"


def test_upload_run_payload_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        requests.Timeout(),
        _response(502),
        _response(201),
    ]
    mocked_post = MagicMock(side_effect=responses)
    monkeypatch.setattr("src.utils.app_client.requests.post", mocked_post)
    monkeypatch.setattr("src.utils.app_client.time.sleep", lambda *_args: None)

    result = upload_run_payload(
        upload_url="https://example.test/api/v1/runs",
        token="token123",
        payload={"schema_version": 1},
    )

    assert result.attempted is True
    assert result.connected is True
    assert result.succeeded is True
    assert result.status_code == 201
    assert mocked_post.call_count == 3


def test_upload_run_payload_does_not_retry_permanent_client_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocked_post = MagicMock(return_value=_response(400))
    monkeypatch.setattr("src.utils.app_client.requests.post", mocked_post)

    result = upload_run_payload(
        upload_url="https://example.test/api/v1/runs",
        token="token123",
        payload={"schema_version": 1},
    )

    assert result.succeeded is False
    assert result.failure_reason == "http_400"
    assert mocked_post.call_count == 1
