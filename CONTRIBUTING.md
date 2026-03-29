# Contributing

## Local Setup

Use Python 3.12 plus `uv` for local development:

```bash
uv sync --frozen --group dev
uv run pytest -q
```

Build the Docker action locally before opening a release-oriented change:

```bash
docker build -t ml-ci-action .
```

## Scope

This repository is intentionally action-first. Keep v0.2.x changes focused on:

- model metric comparison
- tabular data validation
- model card generation
- PR reporting
- documentation and examples

Do not add hosted services, billing flows, or dashboard features here.

## Release Checklist

- `uv run pytest -q` passes locally
- Docker image builds
- the GHCR-backed action manifest is switched back to `Dockerfile` when using `uses: ./` in self-tests
- README examples still match the current action contract
- new inputs or outputs are reflected in `action.yml` and `README.md`
