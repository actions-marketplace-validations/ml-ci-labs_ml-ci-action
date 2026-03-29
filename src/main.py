#!/usr/bin/env python3
"""ML CI Action entry point.

Reads GitHub Action inputs from environment variables, orchestrates
model validation, data validation, model card generation, and PR commenting.
"""

from __future__ import annotations

import json
import os
import sys


def get_input(name: str, required: bool = False, default: str = "") -> str:
    """Read a GitHub Action input from environment.

    Docker actions in GitHub can surface inputs with hyphenated names, while
    local shells generally require underscores. Support both forms.
    """
    normalized = name.upper().replace("-", "_").replace(" ", "_")
    candidates = [
        f"INPUT_{name.upper()}",
        f"INPUT_{normalized}",
    ]

    value = default
    for env_name in candidates:
        if env_name in os.environ:
            value = os.environ[env_name]
            break

    value = value.strip()
    if required and not value:
        print(f"::error::Required input '{name}' is not set")
        sys.exit(1)
    return value


def set_output(name: str, value: str) -> None:
    """Write a single-line output to GITHUB_OUTPUT."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{name}={value}\n")


def set_multiline_output(name: str, value: str) -> None:
    """Write a multiline output to GITHUB_OUTPUT using delimiter syntax."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        delimiter = f"EOF_{os.urandom(8).hex()}"
        with open(output_file, "a") as f:
            f.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")


def resolve_path(path: str) -> str:
    """Resolve a path relative to GITHUB_WORKSPACE."""
    if os.path.isabs(path):
        return path
    workspace = os.environ.get("GITHUB_WORKSPACE", ".")
    return os.path.join(workspace, path)


def emit_outputs(
    validation_passed: bool,
    regression_detected: bool,
    model_card_path: str = "",
    report_data: dict | None = None,
) -> None:
    """Emit all declared action outputs before exiting."""
    set_output("validation-passed", str(validation_passed).lower())
    set_output("regression-detected", str(regression_detected).lower())
    set_output("model-card-path", model_card_path)
    payload = report_data or {
        "validation_passed": validation_passed,
        "regression_detected": regression_detected,
    }
    set_multiline_output("report-json", json.dumps(payload, indent=2))


def main() -> None:
    """Main entry point for the ML CI Action."""
    # --- Read inputs ---
    metrics_file = get_input("METRICS-FILE", required=True)
    baseline_metrics = get_input("BASELINE-METRICS", default="")
    data_path = get_input("DATA-PATH", default="")
    baseline_data_path = get_input("BASELINE-DATA-PATH", default="")
    drift_threshold = float(get_input("DRIFT-THRESHOLD", default="0.1"))
    regression_test = get_input("REGRESSION-TEST", default="threshold")
    regression_tolerance = float(get_input("REGRESSION-TOLERANCE", default="0.02"))
    model_card_enabled = get_input("MODEL-CARD", default="false").lower() == "true"
    fail_on_regression = get_input("FAIL-ON-REGRESSION", default="true").lower() == "true"
    comment_on_pr = get_input("COMMENT-ON-PR", default="true").lower() == "true"
    github_token = get_input("GITHUB-TOKEN", default=os.environ.get("GITHUB_TOKEN", ""))

    # --- Import modules (after input validation to fail fast) ---
    from src.reporters.model_card import generate_model_card
    from src.reporters.pr_comment import (
        generate_report,
        get_pr_number,
        post_or_update_comment,
    )
    from src.utils.metrics import (
        BaselineFetchError,
        MetricsData,
        load_metrics,
        load_metrics_from_github,
    )
    from src.validators.data_validator import validate_data
    from src.validators.model_validator import validate_model

    # --- Load current metrics ---
    metrics_path = resolve_path(metrics_file)
    print(f"::notice::Loading metrics from {metrics_path}")
    try:
        current = load_metrics(metrics_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"::error::Failed to load metrics: {e}")
        emit_outputs(
            validation_passed=False,
            regression_detected=False,
            report_data={
                "validation_passed": False,
                "regression_detected": False,
                "error": str(e),
            },
        )
        sys.exit(1)

    print(f"::notice::Model: {current.model_name} ({current.framework})")
    print(f"::notice::Metrics: {', '.join(current.metrics.keys())}")

    # --- Load baseline metrics ---
    baseline: MetricsData | None = None
    if baseline_metrics:
        if baseline_metrics.lower() in ("main", "master"):
            # Fetch from GitHub API
            repo = os.environ.get("GITHUB_REPOSITORY", "")
            if not repo:
                print("::warning::GITHUB_REPOSITORY not set, cannot fetch baseline from remote")
            elif not github_token:
                print("::warning::GITHUB_TOKEN not provided, cannot fetch baseline from remote")
            else:
                try:
                    baseline = load_metrics_from_github(
                        repo=repo,
                        path=metrics_file,
                        ref=baseline_metrics.lower(),
                        token=github_token,
                    )
                    print(f"::notice::Loaded baseline from {baseline_metrics} branch")
                except FileNotFoundError as e:
                    print(f"::warning::{e}")
                    print("::notice::Proceeding without baseline comparison")
                except BaselineFetchError as e:
                    print(f"::warning::{e}")
                    print(
                        "::notice::Proceeding without baseline comparison. "
                        "Prefer a local file path for 'baseline-metrics' when remote fetches are blocked."
                    )
                except Exception as e:
                    print(f"::warning::Failed to fetch baseline: {e}")
                    print("::notice::Proceeding without baseline comparison")
        else:
            # Load from local file
            baseline_path = resolve_path(baseline_metrics)
            try:
                baseline = load_metrics(baseline_path)
                print(f"::notice::Loaded baseline from {baseline_path}")
            except (FileNotFoundError, ValueError) as e:
                print(f"::warning::Failed to load baseline: {e}")
                print("::notice::Proceeding without baseline comparison")

    # --- Model validation ---
    model_result = None
    if baseline is not None:
        try:
            model_result = validate_model(
                current=current,
                baseline=baseline,
                regression_method=regression_test,
                tolerance=regression_tolerance,
            )
        except ValueError as e:
            print(f"::error::{e}")
            emit_outputs(
                validation_passed=False,
                regression_detected=False,
                report_data={
                    "validation_passed": False,
                    "regression_detected": False,
                    "model_name": current.model_name,
                    "framework": current.framework,
                    "current_metrics": current.metrics,
                    "error": str(e),
                },
            )
            sys.exit(1)
        print(f"::notice::{model_result.summary}")
    else:
        print("::notice::No baseline available — reporting current metrics only")

    # --- Data validation ---
    data_result = None
    if data_path:
        resolved_data = resolve_path(data_path)
        resolved_baseline_data = resolve_path(baseline_data_path) if baseline_data_path else None
        try:
            data_result = validate_data(
                data_path=resolved_data,
                baseline_data_path=resolved_baseline_data,
                drift_threshold=drift_threshold,
            )
            status = "passed" if data_result.overall_passed else "FAILED"
            print(f"::notice::Data validation {status}")
        except Exception as e:
            print(f"::warning::Data validation failed: {e}")

    # --- Model card generation ---
    model_card_path: str | None = None
    if model_card_enabled:
        try:
            output_dir = resolve_path(".")
            model_card_path = generate_model_card(
                metrics_data=current,
                comparisons=model_result.comparisons if model_result else None,
                output_path=os.path.join(output_dir, "MODEL_CARD.md"),
            )
            print(f"::notice::Model card generated at {model_card_path}")
        except Exception as e:
            print(f"::warning::Model card generation failed: {e}")

    # --- PR comment ---
    if comment_on_pr:
        pr_number = get_pr_number()
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        if pr_number and repo and github_token:
            try:
                report = generate_report(
                    model_result=model_result,
                    data_result=data_result,
                    model_card_path=model_card_path,
                )
                post_or_update_comment(
                    repo=repo,
                    pr_number=pr_number,
                    body=report,
                    token=github_token,
                )
            except Exception as e:
                print(f"::warning::Failed to post PR comment: {e}")
        elif not pr_number:
            print("::notice::Not running in a PR context — skipping comment")

    # --- Set outputs ---
    regression_detected = (
        model_result is not None and model_result.regression_result.regression_detected
    )
    data_passed = data_result is None or data_result.overall_passed
    validation_passed = not regression_detected and data_passed

    report_data = {
        "validation_passed": validation_passed,
        "regression_detected": regression_detected,
        "model_name": current.model_name,
        "framework": current.framework,
        "current_metrics": current.metrics,
    }
    if model_result:
        report_data["comparisons"] = [
            {
                "name": c.name,
                "current": c.current,
                "baseline": c.baseline,
                "delta": c.delta,
                "delta_pct": c.delta_pct,
                "improved": c.improved,
                "regression": c.regression,
            }
            for c in model_result.comparisons
        ]
        report_data["regression_test"] = {
            "method": model_result.regression_result.method,
            "detected": model_result.regression_result.regression_detected,
            "details": model_result.regression_result.details,
        }
    emit_outputs(
        validation_passed=validation_passed,
        regression_detected=regression_detected,
        model_card_path=model_card_path or "",
        report_data=report_data,
    )

    # --- Exit with failure if regression detected ---
    if fail_on_regression and regression_detected:
        print("::error::Model regression detected — failing CI")
        sys.exit(1)

    if data_result and not data_result.overall_passed and fail_on_regression:
        print("::error::Data validation failed — failing CI")
        sys.exit(1)

    print("::notice::ML-CI validation complete")


if __name__ == "__main__":
    main()
