"""Helpers for optional ML-CI App connectivity and uploads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import requests

from src.version import VERSION


RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
GITHUB_API_URL = "https://api.github.com/installation/repositories"


@dataclass(frozen=True)
class AppConnectionStatus:
    """Result of probing whether the repo is reachable with the app token."""

    connected: bool
    reason: str | None = None


@dataclass(frozen=True)
class UploadResult:
    """Non-blocking upload attempt status."""

    attempted: bool
    connected: bool
    succeeded: bool
    status_code: int | None = None
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe mapping for report artifacts."""
        return asdict(self)


def detect_app_connection(
    repo: str,
    token: str,
    *,
    timeout: int = 10,
) -> AppConnectionStatus:
    """Return whether the installation token can reach the current repository."""
    if not token:
        return AppConnectionStatus(connected=False, reason="upload_token_missing")
    if not repo:
        return AppConnectionStatus(connected=False, reason="missing_repository")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    repo_normalized = repo.lower()

    try:
        for page in range(1, 11):
            response = requests.get(
                GITHUB_API_URL,
                headers=headers,
                params={"per_page": 100, "page": page},
                timeout=timeout,
            )
            if response.status_code == 401:
                return AppConnectionStatus(connected=False, reason="unauthorized")
            if response.status_code == 403:
                return AppConnectionStatus(connected=False, reason="forbidden")
            if response.status_code == 404:
                return AppConnectionStatus(connected=False, reason="not_found")
            response.raise_for_status()

            repositories = response.json().get("repositories", [])
            if not repositories:
                break
            if any(item.get("full_name", "").lower() == repo_normalized for item in repositories):
                return AppConnectionStatus(connected=True)
    except requests.RequestException:
        return AppConnectionStatus(connected=False, reason="probe_failed")

    return AppConnectionStatus(connected=False, reason="repo_not_accessible")


def upload_run_payload(
    upload_url: str,
    token: str,
    payload: dict[str, object],
    *,
    timeout: int = 10,
    max_attempts: int = 3,
) -> UploadResult:
    """POST a run payload to the configured backend without affecting CI flow."""
    if not upload_url:
        return UploadResult(
            attempted=False,
            connected=bool(token),
            succeeded=False,
            failure_reason="upload_url_missing",
        )
    if not token:
        return UploadResult(
            attempted=False,
            connected=False,
            succeeded=False,
            failure_reason="upload_token_missing",
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": f"ml-ci-action/v{VERSION}",
    }

    last_status_code: int | None = None
    last_reason: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                upload_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            last_status_code = response.status_code
            if response.status_code < 400:
                return UploadResult(
                    attempted=True,
                    connected=True,
                    succeeded=True,
                    status_code=response.status_code,
                )

            last_reason = f"http_{response.status_code}"
            if response.status_code not in RETRYABLE_STATUS_CODES or attempt == max_attempts:
                return UploadResult(
                    attempted=True,
                    connected=True,
                    succeeded=False,
                    status_code=response.status_code,
                    failure_reason=last_reason,
                )
        except requests.Timeout:
            last_reason = "timeout"
            if attempt == max_attempts:
                return UploadResult(
                    attempted=True,
                    connected=True,
                    succeeded=False,
                    status_code=last_status_code,
                    failure_reason=last_reason,
                )
        except requests.ConnectionError:
            last_reason = "connection_error"
            if attempt == max_attempts:
                return UploadResult(
                    attempted=True,
                    connected=True,
                    succeeded=False,
                    status_code=last_status_code,
                    failure_reason=last_reason,
                )
        except requests.RequestException:
            return UploadResult(
                attempted=True,
                connected=True,
                succeeded=False,
                status_code=last_status_code,
                failure_reason="request_error",
            )

        time.sleep(2 ** (attempt - 1))

    return UploadResult(
        attempted=True,
        connected=True,
        succeeded=False,
        status_code=last_status_code,
        failure_reason=last_reason or "upload_failed",
    )
