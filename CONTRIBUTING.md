# Contributing

## Local Setup

Install dependencies in a Python 3.11 environment and run the test suite:

```bash
pip install -r requirements.txt
pytest -q
```

Build the Docker action locally before opening a release-oriented change:

```bash
docker build -t ml-ci-action .
```

## Scope

This repository is intentionally action-first. Keep v0.1 changes focused on:

- model metric comparison
- tabular data validation
- model card generation
- PR reporting
- documentation and examples

Do not add hosted services, billing flows, or dashboard features here.

## Release Checklist

- tests pass locally
- Docker image builds
- README examples still match the current action contract
- new inputs or outputs are reflected in `action.yml` and `README.md`
