import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import anthropic
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import GlobalConfig

logger = logging.getLogger(__name__)


@dataclass
class CompetitionMeta:
    title: str
    task_type: str              # "binary_classification" | "multiclass" | "regression"
    target_column: str
    eval_metric: str
    eval_metric_direction: str  # "maximize" | "minimize"
    id_column: Optional[str]
    feature_notes: list = field(default_factory=list)
    suggested_features: list = field(default_factory=list)
    description_summary: str = ""
    confidence: float = 1.0


_SYSTEM_PROMPT = """\
You are an expert Kaggle data scientist. You will be given a competition description and must extract structured information from it.

Return ONLY a JSON object (no markdown fences, no extra text) matching this exact schema:
{
  "title": "competition title",
  "task_type": "binary_classification | multiclass | regression",
  "target_column": "exact column name to predict",
  "eval_metric": "metric name e.g. roc_auc, rmse, accuracy, logloss, mae, f1",
  "eval_metric_direction": "maximize | minimize",
  "id_column": "id column name or null",
  "feature_notes": ["observation about a specific feature or data quality issue"],
  "suggested_features": ["concrete feature engineering idea"],
  "description_summary": "2-sentence summary of the competition goal",
  "confidence": 0.95
}

Rules:
- task_type must be exactly one of: binary_classification, multiclass, regression
- eval_metric_direction: maximize for accuracy/auc/f1; minimize for rmse/mae/logloss
- feature_notes: list 3-8 observations about the data (NaN patterns, cardinality, date columns, text fields)
- suggested_features: list 3-6 concrete ideas (e.g. "extract title from Name column", "bin Age into decades")
- confidence: your confidence that you parsed the task correctly (0.0-1.0)
"""


class CompetitionReader:
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def fetch_description(self, competition_name: str) -> str:
        import kaggle
        try:
            # Get competition details via Kaggle API
            competitions = kaggle.api.competitions_list(search=competition_name)
            for comp in competitions:
                if comp.ref.lower() == competition_name.lower() or comp.ref.lower().endswith(f"/{competition_name.lower()}"):
                    description = getattr(comp, "description", "") or ""
                    title = getattr(comp, "title", competition_name)
                    evaluation_metric = getattr(comp, "evaluationMetric", "") or ""
                    tags = getattr(comp, "tags", []) or []
                    tag_str = ", ".join(str(t) for t in tags)
                    return (
                        f"Competition: {title}\n"
                        f"Evaluation Metric: {evaluation_metric}\n"
                        f"Tags: {tag_str}\n\n"
                        f"Description:\n{description}"
                    )
        except Exception as e:
            logger.warning(f"Kaggle API description fetch failed: {e}")

        # Fallback: return minimal info so Claude can still try
        return f"Competition slug: {competition_name}\nNo description available via API."

    @retry(
        retry=retry_if_exception(lambda e: isinstance(e, anthropic.APIStatusError) and e.status_code >= 500),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
    )
    def _call_claude(self, description: str) -> str:
        response = self.client.messages.create(
            model=self.config.claude_model,
            max_tokens=1024,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {"role": "user", "content": f"Extract competition info from this description:\n\n{description}"}
            ],
        )
        return response.content[0].text

    def analyze(self, competition_name: str, description: str) -> CompetitionMeta:
        logger.info(f"Analyzing competition '{competition_name}' with Claude...")
        raw = self._call_claude(description)

        # Strip markdown fences if Claude wrapped the JSON anyway
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}\nResponse: {raw}")
            raise ValueError(f"Claude returned unparseable JSON for competition '{competition_name}'") from e

        return CompetitionMeta(
            title=data.get("title", competition_name),
            task_type=data.get("task_type", "binary_classification"),
            target_column=data.get("target_column", ""),
            eval_metric=data.get("eval_metric", ""),
            eval_metric_direction=data.get("eval_metric_direction", "maximize"),
            id_column=data.get("id_column"),
            feature_notes=data.get("feature_notes", []),
            suggested_features=data.get("suggested_features", []),
            description_summary=data.get("description_summary", ""),
            confidence=float(data.get("confidence", 1.0)),
        )

    def load_or_analyze(self, competition_name: str, workspace: Path) -> CompetitionMeta:
        meta_path = workspace / "competition_meta.json"
        if meta_path.exists():
            logger.info("Loading cached competition meta from workspace.")
            data = json.loads(meta_path.read_text())
            return CompetitionMeta(**data)

        description = self.fetch_description(competition_name)
        meta = self.analyze(competition_name, description)
        meta_path.write_text(json.dumps(asdict(meta), indent=2))
        logger.info(f"Competition meta saved to {meta_path}")
        return meta
