# ML-CI Action

Model validation, data quality checks, and model cards for GitHub pull requests, in one Action, with no external dependencies.

![Example ML-CI PR comment](./docs/assets/pr-comment-screenshot.png)

## What It Does

- Catches model regression in PRs. Compares current metrics against `main`, a checked-in artifact, or an explicit remote ref using threshold, Wilcoxon signed-rank, or bootstrap CI tests.
- Validates training data before it merges. Checks schema drift, missing values, duplicate rows, numeric distribution drift, and label balance shifts from CSV or Parquet files.
- Generates model cards automatically from your training output.
- Reports directly in the pull request with one idempotent PR comment, stable markdown and JSON artifacts, or both.
- Runs without a server, account, or external platform. Works from a metrics JSON file.

## Quick Start

```yaml
name: ML Validation
on: [pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Train or evaluate
        run: python train.py --save-metrics metrics.json

      - name: Validate against baseline
        uses: ml-ci-labs/ml-ci-action@v0.4.1
        with:
          metrics-file: metrics.json
          baseline-metrics: main
          regression-test: threshold
          regression-tolerance: "0.02"
          report-mode: comment
          github-token: ${{ github.token }}
```

First PR in a new repo? Drop `baseline-metrics` and ML-CI runs in current-only mode until the metrics file exists on `main`.

## Repo Config

Drop a `.ml-ci.yml` at the repo root. ML-CI auto-discovers it and applies it as the default policy for every workflow.

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
  data:
    missing_threshold: 0.1
    missing_thresholds:
      optional_feature: 0.9
    label_column: target
    exclude_columns:
      - row_id
      - generated_at
```

`fail` metrics block the PR. `warn` metrics appear in the report without failing CI. Workflow inputs override config when explicitly set.

## Statistical Regression Testing

For more rigorous detection than thresholds, add paired `observations` vectors and choose a statistical method:

```yaml
      - uses: ml-ci-labs/ml-ci-action@v0.4.1
        with:
          metrics-file: metrics.json
          baseline-metrics: main
          regression-test: wilcoxon    # or: bootstrap
          alpha: "0.05"
```

| Method | When to use | Detects regression when |
|---|---|---|
| `threshold` | Single evaluation runs | Any metric drops beyond tolerance |
| `wilcoxon` | Cross-validation with paired folds | p-value < `alpha` and direction is bad |
| `bootstrap` | Cross-validation, want confidence bounds | Entire CI falls on the wrong side of zero |

Both statistical methods require paired `observations` vectors in the metrics JSON. See [full reference](./docs/reference.md) for the schema.

## Examples

| Scenario | Workflow |
|---|---|
| First PR, no baseline yet | [pytorch-classification](./examples/pytorch-classification/.github/workflows/ml-ci.yml) |
| Default-branch comparison | [sklearn-regression](./examples/sklearn-regression/.github/workflows/ml-ci.yml) |
| Cross-validation + Wilcoxon | [cross-validation](./examples/cross-validation/.github/workflows/ml-ci.yml) |
| Repo policy (minimal) | [config-minimal](./examples/config-minimal/.github/workflows/ml-ci.yml) |
| Repo policy (advanced) | [config-advanced](./examples/config-advanced/.github/workflows/ml-ci.yml) |
| Artifact-only + downstream consumption | [artifact-consumption](./examples/artifact-consumption/.github/workflows/ml-ci.yml) |

## Docs

| Resource | Description |
|---|---|
| [Full reference](./docs/reference.md) | All inputs, outputs, metrics schema, baseline modes, report modes |
| [Payload schema](./PAYLOAD_SCHEMA.md) | Versioned upload contract for the future ML-CI App |
| [Release checklist](./docs/release-checklist.md) | Release-day verification steps |
| [Contributing](./CONTRIBUTING.md) | Local development workflow |

## Community

- [Report a bug or request a feature](https://github.com/ml-ci-labs/ml-ci-action/issues)
- [Discussions](https://github.com/ml-ci-labs/ml-ci-action/discussions)
- [GitHub Sponsors](https://github.com/sponsors/ml-ci-labs)

## Development

```bash
uv sync --frozen --group dev
uv run pytest -q
docker build -t ml-ci-action .
```
