FROM python:3.12-slim AS builder

LABEL org.opencontainers.image.source="https://github.com/ml-ci-labs/ml-ci-action"
LABEL org.opencontainers.image.description="ML CI/CD GitHub Action — model validation, data quality, model cards"
LABEL org.opencontainers.image.licenses="AGPL-3.0"

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/ml-ci-labs/ml-ci-action"
LABEL org.opencontainers.image.description="ML CI/CD GitHub Action — model validation, data quality, model cards"
LABEL org.opencontainers.image.licenses="AGPL-3.0"

# Git is needed for fallback baseline fetching via git-show.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/

ENV PATH="/app/.venv/bin:${PATH}"
ENTRYPOINT ["python", "-m", "src.main"]
