# Release Checklist

## Automated checks

- `pytest -q`
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

## Manual checks

- README quick start is still copy-paste correct
- the visual asset in `docs/assets/` matches the current PR report format
- rerunning the PR self-test updates the same PR comment instead of creating a duplicate
- after merge to `main`, open one small follow-up PR and verify `baseline-metrics: main` performs a real branch fetch instead of the pre-merge fallback path
- Marketplace metadata matches the README and current stable inputs/outputs

## Release steps

1. Merge the tested branch to `main`.
2. Confirm `main` is green in GitHub Actions.
3. Tag the merge commit as the current release version (currently `v0.2.0`).
4. Publish the action on GitHub Marketplace.
5. Post using the drafts in `docs/launch-copy.md`.
