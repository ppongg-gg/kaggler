import json
import pytest
from unittest.mock import MagicMock, patch
from kaggler.competition_reader import CompetitionReader, CompetitionMeta
from kaggler.config import GlobalConfig
from pathlib import Path


@pytest.fixture
def global_cfg():
    return GlobalConfig(
        kaggle_username="test",
        kaggle_key="test",
        anthropic_api_key="test",
        workspace_root=Path("./workspaces"),
        max_daily_submissions=5,
        claude_model="claude-sonnet-4-6",
    )


@pytest.fixture
def mock_claude_response():
    return json.dumps({
        "title": "Titanic - Machine Learning from Disaster",
        "task_type": "binary_classification",
        "target_column": "Survived",
        "eval_metric": "accuracy",
        "eval_metric_direction": "maximize",
        "id_column": "PassengerId",
        "feature_notes": ["Cabin has many NaNs", "Name contains title"],
        "suggested_features": ["extract title from Name", "bin Age into decades"],
        "description_summary": "Predict survival on the Titanic.",
        "confidence": 0.97,
    })


def test_analyze_parses_valid_response(global_cfg, mock_claude_response, tmp_path):
    reader = CompetitionReader(global_cfg)

    mock_content = MagicMock()
    mock_content.text = mock_claude_response
    mock_response = MagicMock()
    mock_response.content = [mock_content]

    with patch.object(reader.client.messages, "create", return_value=mock_response):
        meta = reader.analyze("titanic", "Predict survival on the Titanic.")

    assert meta.task_type == "binary_classification"
    assert meta.target_column == "Survived"
    assert meta.eval_metric == "accuracy"
    assert meta.id_column == "PassengerId"
    assert meta.confidence == 0.97


def test_analyze_strips_markdown_fences(global_cfg, mock_claude_response, tmp_path):
    reader = CompetitionReader(global_cfg)

    mock_content = MagicMock()
    mock_content.text = f"```json\n{mock_claude_response}\n```"
    mock_response = MagicMock()
    mock_response.content = [mock_content]

    with patch.object(reader.client.messages, "create", return_value=mock_response):
        meta = reader.analyze("titanic", "description")

    assert meta.target_column == "Survived"


def test_load_or_analyze_uses_cache(global_cfg, mock_claude_response, tmp_path):
    reader = CompetitionReader(global_cfg)
    meta_path = tmp_path / "competition_meta.json"
    cached_data = {
        "title": "Cached Title",
        "task_type": "regression",
        "target_column": "SalePrice",
        "eval_metric": "rmse",
        "eval_metric_direction": "minimize",
        "id_column": "Id",
        "feature_notes": [],
        "suggested_features": [],
        "description_summary": "",
        "confidence": 1.0,
    }
    meta_path.write_text(json.dumps(cached_data))

    meta = reader.load_or_analyze("house-prices", tmp_path)
    assert meta.task_type == "regression"
    assert meta.target_column == "SalePrice"
