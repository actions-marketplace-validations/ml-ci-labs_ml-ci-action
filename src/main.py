#!/usr/bin/env python3
"""ML CI Action entry point.

Reads GitHub Action inputs from environment variables, orchestrates
model validation, data validation, model card generation, and PR commenting.
"""

from __future__ import annotations

import json
import os
import sys


def _input_candidates(name: str) -> list[str]:
    """Return supported environment variable names for a GitHub Action input."""
    normalized = name.upper().replace("-", "_").replace(" ", "_")
    return [
        f"INPUT_{name.upper()}",
        f"INPUT_{normalized}",
    ]


def get_input(name: str, required: bool = False, default: str = "") -> str:
    """Read a GitHub Action input from environment.

    Docker actions in GitHub can surface inputs with hyphenated names, while
    local shells generally require underscores. Support both forms.
    """
    value = default
    for env_name in _input_candidates(name):
        if env_name in os.environ:
            value = os.environ[env_name]
            break

    value = value.strip()
    if required and not value:
        print(f"::error::Required input '{name}' is not set")
        sys.exit(1)
    return value


def has_input(name: str) -> bool:
    """Return True when the workflow supplied a specific input."""
    for env_name in _input_candidates(name):
        value = os.environ.get(env_name)
        if value is not None and value.strip() != "":
            return True
    return False


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
    app_connected: bool = False,
    model_card_path: str = "",
    report_markdown_path: str = "",
    report_json_path: str = "",
    report_data: dict | None = None,
) -> None:
    """Emit all declared action outputs before exiting."""
    set_output("validation-passed", str(validation_passed).lower())
    set_output("regression-detected", str(regression_detected).lower())
    set_output("app-connected", str(app_connected).lower())
    set_output("model-card-path", model_card_path)
    set_output("report-markdown-path", report_markdown_path)
    set_output("report-json-path", report_json_path)
    payload = report_data or {
        "validation_passed": validation_passed,
        "regression_detected": regression_detected,
    }
    set_multiline_output("report-json", json.dumps(payload, indent=2))


def write_report_artifacts(workspace: str, markdown_report: str, report_data: dict) -> tuple[str, str]:
    """Persist stable markdown and JSON artifacts for downstream workflow steps."""
    artifact_dir = os.path.join(workspace, ".ml-ci")
    os.makedirs(artifact_dir, exist_ok=True)

    markdown_path = os.path.join(artifact_dir, "validation-report.md")
    json_path = os.path.join(artifact_dir, "validation-report.json")

    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_report)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report_data, handle, indent=2)
        handle.write("\n")

    return markdown_path, json_path


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
    report_mode_raw = get_input("REPORT-MODE", default="")
    baseline_ref = get_input("BASELINE-REF", default="")
    baseline_path = get_input("BASELINE-PATH", default="")
    github_token = get_input("GITHUB-TOKEN", default=os.environ.get("GITHUB_TOKEN", ""))
    framework_hint = get_input("FRAMEWORK", default="auto")
    alpha = float(get_input("ALPHA", default="0.05"))
    n_bootstrap = int(get_input("N-BOOTSTRAP", default="10000"))
    confidence = float(get_input("CONFIDENCE", default="0.95"))
    higher_is_better_raw = get_input("HIGHER-IS-BETTER", default="")
    upload_url = get_input("UPLOAD-URL", default="")
    upload_token = get_input("UPLOAD-TOKEN", default="")

    # --- Parse higher-is-better overrides ---
    higher_is_better: dict[str, bool] | None = None
    if higher_is_better_raw:
        try:
            higher_is_better = json.loads(higher_is_better_raw)
            if not isinstance(higher_is_better, dict):
                raise ValueError("must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"::error::Invalid 'higher-is-better' input: {e}")
            sys.exit(1)

    explicit_regression_test = regression_test if has_input("REGRESSION-TEST") else None
    explicit_regression_tolerance = (
        regression_tolerance if has_input("REGRESSION-TOLERANCE") else None
    )
    explicit_higher_is_better = higher_is_better if has_input("HIGHER-IS-BETTER") else {}

    # --- Import modules (after input validation to fail fast) ---
    from src.reporters.model_card import generate_model_card
    from src.reporters.pr_comment import (
        generate_report,
        get_pr_number,
        post_or_update_comment,
    )
    from src.utils.app_client import (
        UploadResult,
        detect_app_connection,
        upload_run_payload,
    )
    from src.utils.app_payload import build_run_payload
    from src.utils.metrics import (
        BaselineSource,
        BaselineFetchError,
        load_metrics,
        load_metrics_from_github,
        resolve_baseline_source,
    )
    from src.utils.policy import (
        PolicyConfigError,
        WorkflowPolicyOverrides,
        discover_policy_file,
        load_policy_config,
        resolve_policy,
    )
    from src.validators.data_validator import validate_data
    from src.validators.model_validator import validate_model

    workspace = os.environ.get("GITHUB_WORKSPACE", ".")

    explicit_report_mode = has_input("REPORT-MODE")
    if explicit_report_mode:
        report_mode = report_mode_raw.lower()
        if report_mode not in {"comment", "artifact", "both"}:
            print(
                "::error::Invalid 'report-mode'. Expected one of: comment, artifact, both"
            )
            emit_outputs(
                validation_passed=False,
                regression_detected=False,
                report_data={
                    "validation_passed": False,
                    "regression_detected": False,
                    "error": "Invalid report-mode",
                },
            )
            sys.exit(1)
        comment_enabled = report_mode in {"comment", "both"}
        artifact_enabled = report_mode in {"artifact", "both"}
        if comment_on_pr != comment_enabled or artifact_enabled:
            print("::notice::'report-mode' takes precedence over legacy 'comment-on-pr'")
    else:
        comment_enabled = comment_on_pr
        artifact_enabled = False
        if not comment_on_pr:
            print(
                "::notice::PR comments disabled via legacy 'comment-on-pr'. "
                "Use 'report-mode: artifact' or 'both' to emit file artifacts."
            )

    config = None
    policy_path = discover_policy_file(workspace)
    if policy_path is not None:
        try:
            config = load_policy_config(policy_path)
            print(f"::notice::Loaded repo policy from {policy_path}")
        except PolicyConfigError as e:
            print(f"::error::{e}")
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

    policy = resolve_policy(
        config=config,
        overrides=WorkflowPolicyOverrides(
            regression_test=explicit_regression_test,
            regression_tolerance=explicit_regression_tolerance,
            higher_is_better=explicit_higher_is_better or {},
        ),
    )
    print(
        "::notice::Effective regression policy: "
        f"test={policy.regression_test}, tolerance={policy.regression_tolerance:.1%}"
    )
    print(
        "::notice::Effective data policy: "
        f"missing_threshold={policy.data_policy.missing_threshold:.1%}, "
        f"label_column={policy.data_policy.label_column or 'none'}, "
        f"include_columns={policy.data_policy.include_columns or 'all'}, "
        f"exclude_columns={policy.data_policy.exclude_columns or 'none'}"
    )

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

    if framework_hint.lower() != "auto" and current.framework.lower() in ("", "unknown"):
        current.framework = framework_hint
        print(f"::notice::Using framework input fallback: {current.framework}")

    print(f"::notice::Model: {current.model_name} ({current.framework})")
    print(f"::notice::Metrics: {', '.join(current.metrics.keys())}")

    # --- Load baseline metrics ---
    baseline: MetricsData | None = None
    try:
        baseline_source = resolve_baseline_source(
            metrics_file=metrics_file,
            baseline_metrics=baseline_metrics,
            baseline_ref=baseline_ref,
            baseline_path=baseline_path,
        )
    except ValueError as e:
        print(f"::error::{e}")
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

    if baseline_source.mode == "remote-explicit" and baseline_metrics.lower() in {"main", "master"}:
        print(
            "::notice::Using explicit remote baseline inputs. "
            "Ignoring legacy 'baseline-metrics' branch shortcut."
        )

    if baseline_source.mode in {"remote-explicit", "remote-legacy"}:
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        if not repo:
            print("::warning::GITHUB_REPOSITORY not set, cannot fetch baseline from remote")
            baseline_source = BaselineSource(
                **{**baseline_source.__dict__, "reason": "missing_repository"}
            )
        elif not github_token:
            print("::warning::GITHUB_TOKEN not provided, cannot fetch baseline from remote")
            baseline_source = BaselineSource(
                **{**baseline_source.__dict__, "reason": "missing_token"}
            )
        else:
            try:
                baseline = load_metrics_from_github(
                    repo=repo,
                    path=baseline_source.resolved_path or metrics_file,
                    ref=baseline_source.resolved_ref or "",
                    token=github_token,
                )
                baseline_source = BaselineSource(
                    **{**baseline_source.__dict__, "available": True}
                )
                print(
                    f"::notice::Loaded remote baseline from "
                    f"{baseline_source.resolved_ref}:{baseline_source.resolved_path}"
                )
            except FileNotFoundError as e:
                print(f"::warning::{e}")
                print("::notice::Proceeding without baseline comparison")
                baseline_source = BaselineSource(
                    **{**baseline_source.__dict__, "reason": "not_found"}
                )
            except BaselineFetchError as e:
                print(f"::warning::{e}")
                print(
                    "::notice::Proceeding without baseline comparison. "
                    "Prefer a local file path for 'baseline-metrics' when remote fetches are blocked."
                )
                baseline_source = BaselineSource(
                    **{**baseline_source.__dict__, "reason": "fetch_blocked"}
                )
            except Exception as e:
                print(f"::warning::Failed to fetch baseline: {e}")
                print("::notice::Proceeding without baseline comparison")
                baseline_source = BaselineSource(
                    **{**baseline_source.__dict__, "reason": "fetch_failed"}
                )
    elif baseline_source.mode == "local":
        resolved_baseline_path = resolve_path(baseline_source.resolved_path or baseline_metrics)
        try:
            baseline = load_metrics(resolved_baseline_path)
            baseline_source = BaselineSource(
                **{
                    **baseline_source.__dict__,
                    "available": True,
                    "resolved_path": resolved_baseline_path,
                }
            )
            print(f"::notice::Loaded baseline from {resolved_baseline_path}")
        except (FileNotFoundError, ValueError) as e:
            print(f"::warning::Failed to load baseline: {e}")
            print("::notice::Proceeding without baseline comparison")
            baseline_source = BaselineSource(
                **{
                    **baseline_source.__dict__,
                    "resolved_path": resolved_baseline_path,
                    "reason": "load_failed",
                }
            )

    # --- Model validation ---
    model_result = None
    if baseline is not None:
        try:
            model_result = validate_model(
                current=current,
                baseline=baseline,
                regression_method=policy.regression_test,
                tolerance=policy.regression_tolerance,
                higher_is_better=policy.higher_is_better,
                metric_tolerances=policy.metric_tolerances,
                metric_severities=policy.metric_severities,
                alpha=alpha,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
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

    if baseline is not None:
        shared_metrics = sorted(set(current.metrics.keys()) & set(baseline.metrics.keys()))
        current_only_metrics = sorted(set(current.metrics.keys()) - set(baseline.metrics.keys()))
        baseline_only_metrics = sorted(set(baseline.metrics.keys()) - set(current.metrics.keys()))
    else:
        shared_metrics = []
        current_only_metrics = sorted(current.metrics.keys())
        baseline_only_metrics = []

    if shared_metrics:
        print(f"::notice::Shared metrics: {', '.join(shared_metrics)}")
    if current_only_metrics:
        print(f"::notice::Current-only metrics: {', '.join(current_only_metrics)}")
    if baseline_only_metrics:
        print(f"::notice::Baseline-only metrics: {', '.join(baseline_only_metrics)}")

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
                label_column=policy.data_policy.label_column,
                missing_threshold=policy.data_policy.missing_threshold,
                missing_thresholds=policy.data_policy.missing_thresholds,
                include_columns=policy.data_policy.include_columns,
                exclude_columns=policy.data_policy.exclude_columns,
            )
            status = "passed" if data_result.overall_passed else "FAILED"
            print(
                f"::notice::Data validation {status} "
                f"({data_result.failure_count} failure(s), {data_result.warning_count} warning(s))"
            )
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

    report_data = {
        "validation_passed": False,
        "regression_detected": False,
        "blocking_regression_detected": False,
        "model_name": current.model_name,
        "framework": current.framework,
        "current_metrics": current.metrics,
        "shared_metrics": shared_metrics,
        "current_only_metrics": current_only_metrics,
        "baseline_only_metrics": baseline_only_metrics,
        "baseline_source": {
            "mode": baseline_source.mode,
            "requested_ref": baseline_source.requested_ref,
            "requested_path": baseline_source.requested_path,
            "resolved_ref": baseline_source.resolved_ref,
            "resolved_path": baseline_source.resolved_path,
            "available": baseline_source.available,
            "reason": baseline_source.reason,
        },
        "data_policy": {
            "missing_threshold": policy.data_policy.missing_threshold,
            "missing_thresholds": policy.data_policy.missing_thresholds,
            "label_column": policy.data_policy.label_column,
            "include_columns": policy.data_policy.include_columns,
            "exclude_columns": policy.data_policy.exclude_columns,
        },
    }

    regression_detected = model_result is not None and model_result.regression_result.regression_detected
    blocking_regression_detected = (
        model_result is not None and model_result.blocking_regression_count > 0
    )
    data_passed = data_result is None or data_result.overall_passed
    validation_passed = not blocking_regression_detected and data_passed

    report_data["validation_passed"] = validation_passed
    report_data["regression_detected"] = regression_detected
    report_data["blocking_regression_detected"] = blocking_regression_detected
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
                "severity": c.severity,
                "tolerance": c.tolerance,
            }
            for c in model_result.comparisons
        ]
        report_data["regression_test"] = {
            "method": model_result.regression_result.method,
            "detected": model_result.regression_result.regression_detected,
            "blocking_detected": blocking_regression_detected,
            "details": model_result.regression_result.details,
        }
    if data_result:
        report_data["data_validation"] = {
            "schema_valid": data_result.schema_valid,
            "schema_errors": data_result.schema_errors,
            "schema_warnings": data_result.schema_warnings,
            "missing_value_report": data_result.missing_value_report,
            "missing_value_failures": data_result.missing_value_failures,
            "missing_value_thresholds": data_result.missing_value_thresholds,
            "duplicate_count": data_result.duplicate_count,
            "duplicate_pct": data_result.duplicate_pct,
            "label_column": data_result.label_column,
            "label_distribution": data_result.label_distribution,
            "baseline_label_distribution": data_result.baseline_label_distribution,
            "label_distribution_shift": data_result.label_distribution_shift,
            "label_shift_detected": data_result.label_shift_detected,
            "drift_scores": data_result.drift_scores,
            "drift_detected": data_result.drift_detected,
            "filtered_columns": data_result.filtered_columns,
            "warnings": data_result.warnings,
            "failures": data_result.failures,
            "details": data_result.details,
        }

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    connection_status = detect_app_connection(repo=repo, token=upload_token)
    app_connected = connection_status.connected
    report_data["app_connected"] = app_connected

    upload_result = UploadResult(
        attempted=False,
        connected=app_connected,
        succeeded=False,
        failure_reason=None,
    )
    if upload_url and upload_token and app_connected:
        payload = build_run_payload(
            current=current,
            report_data=report_data,
            model_card_path=model_card_path,
        )
        upload_result = upload_run_payload(
            upload_url=upload_url,
            token=upload_token,
            payload=payload,
        )
        if upload_result.succeeded:
            print("::notice::Uploaded ML-CI run payload")
        else:
            print(
                "::warning::Failed to upload ML-CI run payload: "
                f"{upload_result.failure_reason or 'upload_failed'}"
            )
    elif upload_url and not upload_token:
        upload_result = UploadResult(
            attempted=False,
            connected=False,
            succeeded=False,
            failure_reason="upload_token_missing",
        )
    elif upload_token and not upload_url:
        upload_result = UploadResult(
            attempted=False,
            connected=app_connected,
            succeeded=False,
            failure_reason="upload_url_missing",
        )
    elif upload_url and upload_token and not app_connected:
        upload_result = UploadResult(
            attempted=False,
            connected=False,
            succeeded=False,
            failure_reason=connection_status.reason or "repo_not_accessible",
        )

    if upload_token and not app_connected and connection_status.reason:
        print(
            "::notice::ML-CI App token could not reach this repository: "
            f"{connection_status.reason}"
        )

    report_data["upload"] = upload_result.as_dict()

    markdown_report = generate_report(
        model_result=model_result,
        data_result=data_result,
        model_card_path=model_card_path,
        current_metrics=current.metrics,
        current_only_metrics=current_only_metrics,
        baseline_only_metrics=baseline_only_metrics,
        baseline_source=report_data["baseline_source"],
    )

    report_markdown_path = ""
    report_json_path = ""
    if artifact_enabled:
        report_markdown_path, report_json_path = write_report_artifacts(
            workspace=workspace,
            markdown_report=markdown_report,
            report_data=report_data,
        )
        print(f"::notice::Report artifacts written to {report_markdown_path} and {report_json_path}")

    # --- PR comment ---
    if comment_enabled:
        pr_number = get_pr_number()
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        if pr_number and repo and github_token:
            try:
                post_or_update_comment(
                    repo=repo,
                    pr_number=pr_number,
                    body=markdown_report,
                    token=github_token,
                )
            except Exception as e:
                print(f"::warning::Failed to post PR comment: {e}")
        elif not pr_number:
            print("::notice::Not running in a PR context — skipping comment")

    # --- Set outputs ---
    emit_outputs(
        validation_passed=validation_passed,
        regression_detected=regression_detected,
        app_connected=app_connected,
        model_card_path=model_card_path or "",
        report_markdown_path=report_markdown_path,
        report_json_path=report_json_path,
        report_data=report_data,
    )

    # --- Exit with failure if regression detected ---
    if fail_on_regression and blocking_regression_detected:
        print("::error::Model regression detected — failing CI")
        sys.exit(1)

    if data_result and not data_result.overall_passed and fail_on_regression:
        print("::error::Data validation failed — failing CI")
        sys.exit(1)

    print("::notice::ML-CI validation complete")


if __name__ == "__main__":
    main()
