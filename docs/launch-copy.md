# Launch Copy

## Show HN draft

**Title**

Show HN: ML-CI Action, ML CI/CD without platform lock-in

**Body**

I built an open-source GitHub Action for ML validation in pull requests.

It compares model metrics against a baseline, checks tabular data quality, generates a model card, and posts a PR report with a single `uses:` step. It does not require a hosted ML platform, external server, or experiment-tracking account.

The current v0.1 scope is intentionally narrow:

- threshold-based regression gating
- CSV/Parquet data checks
- model card generation
- idempotent PR comments on GitHub

It is for teams that want GitHub-native ML validation, not a full experiment platform.

Repo: https://github.com/ml-ci-labs/ml-ci-action

## Reddit draft: r/mlops

I built an open-source GitHub Action for ML CI/CD without platform lock-in.

It does four things in PRs:

- compares model metrics to a baseline
- checks tabular data quality
- generates a model card
- posts a PR report and can fail CI on regression

No external service is required. You just give it a metrics JSON file and optional data path.

Current scope is deliberately v0.1:

- threshold-only regression detection
- CSV/Parquet data validation
- GitHub-native PR comments

Repo: https://github.com/ml-ci-labs/ml-ci-action

Happy to get feedback from teams doing ML validation in Actions today.

## Reddit draft: r/MachineLearning

I built an open-source GitHub Action for validating ML changes in GitHub pull requests.

The goal is simple: if a PR changes model code or training logic, the action can compare metrics against baseline, flag regressions, run basic tabular data checks, generate a model card, and post the report directly in the PR.

It is intentionally not a full MLOps platform. It is a lightweight GitHub-native CI step for teams that want reproducible ML checks in code review.

Repo: https://github.com/ml-ci-labs/ml-ci-action

Would especially appreciate feedback on the metrics payload shape and the PR review UX.

## Objection handling

### How is this different from ClearML, W&B, or MLflow?

Those tools are stronger as full platforms. ML-CI is for teams that want GitHub-native validation and PR reporting without standing up or adopting an external platform first.

### Why threshold-only?

It keeps v0.1 understandable and predictable. Threshold gating solves the most common regression problem today. Statistical tests can come later once the core UX is stable.

### Why a Docker action?

The user is usually already in Python/ML tooling, and Docker gives a consistent runtime for Python dependencies and GitHub-hosted runners.

### What data types and frameworks are supported today?

Metrics comparison is framework-agnostic as long as the metrics JSON is valid. The docs and examples are aimed at PyTorch and scikit-learn. Data validation is currently tabular only and supports CSV/Parquet.

## Launch-day metrics

- GitHub stars
- Marketplace installs
- opened issues
- PR comments generated
- repeat users / inbound team interest
