FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/ml-ci-labs/ml-ci-action"
LABEL org.opencontainers.image.description="ML CI/CD GitHub Action — model validation, data quality, model cards"
LABEL org.opencontainers.image.licenses="AGPL-3.0"

# Git is needed for fallback baseline fetching via git-show
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/

WORKDIR /app
ENTRYPOINT ["python", "-m", "src.main"]
