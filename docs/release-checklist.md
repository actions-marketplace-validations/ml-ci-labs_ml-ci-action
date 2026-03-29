# Release Checklist

## Automated checks

- `uv run pytest -q`
- `docker build -t ml-ci-action .`
- container smoke test with fixture metrics
- `.github/workflows/ci.yml` is green on the release branch
- `.github/workflows/self-test-pr.yml` is green on the release PR
  - `current-only-pass`
  - `remote-main-baseline-fallback-pass`
  - `pr-comment-pass`
  - `wilcoxon-pass`
  - `regression-expected-failure`
  - `data-quality-expected-failure`
- `.github/workflows/release.yml` publishes and smoke-tests the GHCR image for the tag

## Manual checks

- README quick start is still copy-paste correct
- the visual asset in `docs/assets/` matches the current PR report format
- rerunning the PR self-test updates the same PR comment instead of creating a duplicate
- after merge to `main`, open one small follow-up PR and verify `baseline-metrics: main` performs a real branch fetch instead of the pre-merge fallback path
- Marketplace metadata matches the README and current stable inputs/outputs

## Marketplace metadata

- Listing title: `ML CI - Model Validation`
- Short description: `GitHub-native ML validation for pull requests: compare metrics, check tabular data, generate model cards, and post one idempotent PR report.`
- Suggested About text: `GitHub-native ML validation for pull requests: compare metrics, check tabular data, generate model cards, and post one idempotent PR report.`
- Suggested topics: `mlops`, `machine-learning`, `github-actions`, `ci-cd`, `model-validation`, `data-quality`, `model-cards`, `pytorch`, `scikit-learn`, `ml-ci`

## Release steps

1. Merge the tested branch to `main`.
2. Confirm `main` is green in GitHub Actions.
3. Tag the merge commit as the current patch release on the `v0.2.x` line.
4. Confirm `.github/workflows/release.yml` pushed `v0.2.x`, `v0.2`, and `v0` image tags to GHCR.
5. Publish or update the action on GitHub Marketplace.
