# ML-CI Reference

This document keeps the full usage and configuration details that are too dense for the top-level README.

## Metrics JSON Schema

`metrics-file` must point to a JSON object with a required `metrics` field.

```json
{
  "model_name": "fraud-detector-v2",
  "framework": "pytorch",
  "timestamp": "2026-03-28T10:30:00Z",
  "metrics": {
    "accuracy": 0.943,
    "f1": 0.891,
    "auc_roc": 0.967,
    "loss": 0.153
  },
  "dataset": {
    "name": "transactions-q1-2026",
    "version": "2026.03",
    "num_samples": 150000
  },
  "hyperparameters": {
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32
  },
  "lineage": {
    "training_data_hash": "sha256:2c26b46b68ffc68ff99b453c1d30413413422b7e5f5aa0d41cf2f4e32cc4c43f",
    "model_artifact_hash": "sha256:fcde2b2edba56bf408601fb721fe9b5c338d10ee429ea04f",
    "environment": {
      "python": "3.11.11",
      "platform": "ubuntu-latest",
      "cuda": "12.4"
    }
  }
}
```

Rules:

- The top level must be a JSON object.
- `metrics` is required.
- Metric values must be numeric scalars.
- `lineage` is optional. When present, `training_data_hash` and `model_artifact_hash` must be strings and `environment` must be an object.
- Comparisons use only the intersection of metric names in current and baseline payloads.
- Metrics with names like `loss`, `mse`, and `error` are treated as lower-is-better by default.

## Baseline Modes

ML-CI supports four baseline modes:

- Local file path: compare against a checked-in artifact or prior run output.
- `main` or `master`: fetch the same metrics path from the named branch via the GitHub Contents API.
- `baseline-ref` with optional `baseline-path`: fetch a remote baseline from an explicit ref and path.
- Empty: skip baseline comparison and report current metrics only.

If the baseline file is missing on the branch, the action degrades gracefully to current-only reporting.

Explicit remote override example:

```yaml
      - name: Compare against a release baseline artifact
        uses: ml-ci-labs/ml-ci-action@v0.4.1
        with:
          metrics-file: metrics.json
          baseline-ref: release/2026-03-15
          baseline-path: baselines/fraud/latest.json
          report-mode: both
          github-token: ${{ github.token }}
```

## Statistical Regression Testing

For more rigorous regression detection than simple thresholds, add paired observation vectors from cross-validation:

```json
{
  "model_name": "fraud-detector-v2",
  "framework": "pytorch",
  "metrics": {
    "accuracy": 0.943,
    "f1": 0.891
  },
  "observations": {
    "accuracy": [0.94, 0.95, 0.93, 0.94, 0.96, 0.93, 0.95, 0.94, 0.93, 0.95],
    "f1": [0.88, 0.90, 0.89, 0.88, 0.91, 0.88, 0.90, 0.89, 0.88, 0.90]
  }
}
```

Then choose a method:

```yaml
      - name: Wilcoxon signed-rank test
        uses: ml-ci-labs/ml-ci-action@v0.4.1
        with:
          metrics-file: metrics.json
          baseline-metrics: main
          regression-test: wilcoxon
          alpha: "0.05"
```

```yaml
      - name: Bootstrap confidence intervals
        uses: ml-ci-labs/ml-ci-action@v0.4.1
        with:
          metrics-file: metrics.json
          baseline-metrics: main
          regression-test: bootstrap
          confidence: "0.95"
          n-bootstrap: "10000"
```

Both methods require `observations` in both current and baseline metrics files. The observation vectors must be the same length. At least 6 observations are recommended for Wilcoxon.

| Method | When to use | Regression detected when |
|---|---|---|
| `threshold` | Quick checks, single evaluation runs | Any metric drops by more than `regression-tolerance` |
| `wilcoxon` | Cross-validation with paired folds | Wilcoxon p-value < `alpha` and change is in the bad direction |
| `bootstrap` | Cross-validation, want confidence bounds | Entire CI is on the bad side of zero |

## Repo Policy Config

ML-CI supports an optional repo-root policy file at `.ml-ci.yml`.

Minimal example:

```yaml
version: 1
policy:
  regression_test: threshold
  regression_tolerance: 0.02
  metrics:
    accuracy:
      tolerance: 0.01
      direction: higher
      severity: fail
    loss:
      direction: lower
      severity: warn
```

Advanced example:

```yaml
version: 1
policy:
  regression_test: bootstrap
  regression_tolerance: 0.03
  higher_is_better:
    custom_cost: false
  metrics:
    ndcg:
      tolerance: 0.01
      direction: higher
      severity: fail
    loss:
      tolerance: 0.05
      direction: lower
      severity: warn
  data:
    missing_threshold: 0.2
    missing_thresholds:
      optional_feature: 0.8
    label_column: label
    include_columns:
      - feature_a
      - feature_b
      - label
    exclude_columns:
      - row_id
      - generated_at
```

Precedence rules:

- `.ml-ci.yml` defines repo defaults.
- Workflow inputs override config when they are explicitly set.
- Workflow inputs only override equivalent knobs: `regression-test`, `regression-tolerance`, and `higher-is-better`.
- `fail` metrics block CI when they regress. `warn` metrics stay visible in the report without failing CI.
- `policy.data.missing_threshold` sets the default missing-value limit for validated columns.
- `policy.data.missing_thresholds.<column>` overrides the default for specific columns.
- `policy.data.label_column` surfaces class balance and warns when label share shifts materially against a baseline dataset.
- `policy.data.include_columns` and `policy.data.exclude_columns` scope data checks to the columns that matter in production.
- Invalid config fails fast with an actionable error message that includes the config path.

## Inputs

| Input | Required | Default | Description |
|---|---|---:|---|
| `metrics-file` | Yes |  | Path to current metrics JSON. |
| `baseline-metrics` | No | `""` | Local path to baseline metrics, or `main` / `master` to fetch the same path from the default branch. |
| `baseline-ref` | No | `""` | Explicit remote ref for baseline fetches. Activates explicit remote baseline mode. |
| `baseline-path` | No | `""` | Explicit remote path for baseline fetches. Requires `baseline-ref` and defaults to `metrics-file`. |
| `data-path` | No | `""` | Path to CSV or Parquet data for tabular quality checks. |
| `baseline-data-path` | No | `""` | Path to baseline data for schema and drift comparisons. |
| `drift-threshold` | No | `0.1` | Maximum PSI score before flagging drift. |
| `regression-test` | No | `threshold` | Regression method: `threshold`, `wilcoxon`, or `bootstrap`. |
| `regression-tolerance` | No | `0.02` | Maximum allowed per-metric degradation as a fraction of baseline. |
| `alpha` | No | `0.05` | Significance level for Wilcoxon test. |
| `n-bootstrap` | No | `10000` | Number of bootstrap resamples. |
| `confidence` | No | `0.95` | Confidence level for bootstrap CI. |
| `higher-is-better` | No | `""` | JSON object mapping metric names to direction (`true` = higher is better). |
| `model-card` | No | `false` | Generate `MODEL_CARD.md`. |
| `fail-on-regression` | No | `true` | Exit non-zero when blocking regression or data-quality failure is detected. |
| `comment-on-pr` | No | `true` | Legacy PR-comment toggle used only when `report-mode` is unset. |
| `report-mode` | No | `comment` | Report delivery mode: `comment`, `artifact`, or `both`. |
| `framework` | No | `auto` | Framework hint used only when the metrics file omits framework metadata or reports `unknown`. |
| `github-token` | No | `${{ github.token }}` | Token used for PR comments and remote baseline fetches. |
| `upload-url` | No | `""` | Optional ML-CI App ingest endpoint. When unset, upload is skipped. |
| `upload-token` | No | `""` | Optional ML-CI App installation token used for install detection and upload auth. |

## Outputs

| Output | Description |
|---|---|
| `validation-passed` | `true` when no blocking regression or failing data issue was detected. |
| `regression-detected` | `true` when any shared metric regressed, including `warn`-severity metrics. |
| `app-connected` | `true` when the provided ML-CI App token can reach the current repository. |
| `model-card-path` | Output path for the generated model card, if enabled. |
| `report-markdown-path` | Workspace-relative output path for `.ml-ci/validation-report.md` when `report-mode` includes `artifact`. |
| `report-json-path` | Workspace-relative output path for `.ml-ci/validation-report.json` when `report-mode` includes `artifact`. |
| `report-json` | Full JSON payload describing the validation result. |

## Optional App Upload

`upload-url` and `upload-token` are both optional. The Action remains fully useful when neither is set.

- `app-connected` is computed by probing the GitHub App installation token against repository access.
- Upload runs only when both inputs are set and the probe confirms repository access.
- Upload failures are non-blocking and are recorded under the `upload` key in `report-json`.
- The upload payload is versioned separately from `report-json`; see [PAYLOAD_SCHEMA.md](../PAYLOAD_SCHEMA.md).

## Report Modes

`report-mode` controls where the human-readable report goes:

- `comment`: post or update the PR comment only
- `artifact`: write `.ml-ci/validation-report.md` and `.ml-ci/validation-report.json` only
- `both`: do both

If you still use `comment-on-pr`, that legacy toggle is honored only when `report-mode` is unset.

Example artifact-only workflow:

```yaml
      - name: Run ML-CI and emit artifacts
        id: ml_ci
        uses: ml-ci-labs/ml-ci-action@v0.4.1
        with:
          metrics-file: metrics.json
          report-mode: artifact

      - name: Consume the JSON artifact
        run: |
          python - <<'PY'
          import json
          from pathlib import Path

          report = json.loads(Path("${{ steps.ml_ci.outputs.report-json-path }}").read_text())
          print(report["validation_passed"])
          PY
```

## What Shows Up In The PR

Each run creates or updates one idempotent comment with the following sections:

| Section | What it shows |
|---|---|
| `Model Performance` | Baseline vs current metrics, absolute and percentage deltas, and per-metric pass, warn, or fail status |
| `Statistical Details` | Wilcoxon p-values or bootstrap CI bounds in a collapsible block when using statistical tests |
| `Data Quality` | Schema checks, missing-value failures, duplicate warnings, drift findings, and label balance notes |
| `Model Card` | The generated `MODEL_CARD.md` path when model card output is enabled |
| `Summary` | A compact gate result for model regression and data quality |

<details>
<summary>Raw comment markdown example (threshold)</summary>

```markdown
<!-- ml-ci-report -->

## :white_check_mark: ML-CI Validation Report

### Model Performance

**Model**: `fraud-detector-v2` (pytorch)

| Metric | Baseline | Current | Delta | Delta % | Status |
|--------|----------|---------|-------|---------|--------|
| accuracy | 0.9300 | 0.9500 | +0.0200 | +2.2% | :white_check_mark: |
| f1_score | 0.9100 | 0.9300 | +0.0200 | +2.2% | :white_check_mark: |
| loss | 0.1800 | 0.1500 | -0.0300 | -16.7% | :white_check_mark: |

**Regression test**: `threshold` (tolerance: 2.0%)
**Result**: :white_check_mark: All metrics within tolerance

### Summary

- All 3 metrics within tolerance (3 improved)

---
*Generated by [ML-CI](https://github.com/ml-ci-labs/ml-ci-action)*
```

</details>

## Data Validation

When `data-path` is provided, ML-CI performs tabular checks:

- schema compatibility against `baseline-data-path`
- missing-value analysis
- duplicate-row detection
- numeric distribution drift via PSI
- optional label distribution reporting via `policy.data.label_column`

Behavior notes:

- Missing-value failures are blocking.
- Duplicates, drift, new columns, and label-balance shifts are warnings.
- Schema errors tell you what changed and suggest concrete fixes such as excluding noisy columns or refreshing the baseline.
- With zero config, ML-CI validates all columns, tells you when baseline-only checks were skipped, and suggests a likely label column when it finds one.

CSV and Parquet are supported.

## Model Cards

When `model-card: "true"` is enabled, ML-CI writes `MODEL_CARD.md` using the current metrics payload plus optional comparison data.
