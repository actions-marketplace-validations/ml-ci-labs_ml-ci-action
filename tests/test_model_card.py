"""Tests for src.reporters.model_card."""

from pathlib import Path

from src.reporters.model_card import PROJECT_URL, generate_model_card
from src.utils.metrics import MetricsData


def test_model_card_footer_uses_canonical_repo_url(tmp_path: Path):
    metrics_data = MetricsData(
        model_name="test-model",
        framework="pytorch",
        timestamp="2026-03-28T10:30:00Z",
        metrics={"accuracy": 0.95, "loss": 0.15},
        dataset={},
        hyperparameters={},
    )

    output_path = tmp_path / "MODEL_CARD.md"
    generate_model_card(metrics_data, output_path=str(output_path))

    content = output_path.read_text()
    assert PROJECT_URL in content
    assert "github.com/ml-ci/ml-ci-action" not in content
