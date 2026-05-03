import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import GlobalConfig
from .competition_reader import CompetitionMeta
from .tracker import IterationSummary

logger = logging.getLogger(__name__)


@dataclass
class StrategyAdvice:
    summary: str
    next_preset_override: Optional[str] = None
    feature_engineering_additions: list = field(default_factory=list)
    drop_features: list = field(default_factory=list)
    autogluon_extra_args: dict = field(default_factory=dict)
    should_stop: bool = False
    stop_reason: Optional[str] = None

    def to_trainer_hints(self) -> dict:
        return {
            "next_preset_override": self.next_preset_override,
            "autogluon_extra_args": self.autogluon_extra_args,
        }


_SYSTEM_PROMPT = """\
You are an expert Kaggle data scientist advising an automated ML pipeline.

You will receive the competition goal, score history, and AutoGluon's internal model rankings. Your job is to recommend what to try in the next iteration.

Return ONLY a JSON object (no markdown fences, no extra text):
{
  "summary": "brief explanation of recommendation",
  "next_preset_override": null or "medium_quality|good_quality|high_quality|best_quality",
  "feature_engineering_additions": ["concrete feature idea"],
  "drop_features": ["column name to drop if hurting performance"],
  "autogluon_extra_args": {},
  "should_stop": false,
  "stop_reason": null
}

Guidelines:
- next_preset_override: only set if you want to deviate from the escalation schedule
- feature_engineering_additions: concrete actionable ideas only (e.g. "extract title from Name")
- autogluon_extra_args: valid AutoGluon .fit() kwargs (e.g. {"num_stack_levels": 3, "num_bag_folds": 8})
- should_stop: true only if: (a) score has not improved in 3+ iterations AND we're on best_quality, OR (b) only 1 submission slot remains and score is already strong
- stop_reason: required if should_stop is true
"""


class StrategyAdvisor:
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    @retry(
        retry=retry_if_exception(lambda e: isinstance(e, anthropic.APIStatusError) and e.status_code >= 500),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
    )
    def advise(
        self,
        meta: CompetitionMeta,
        history: list[IterationSummary],
        autogluon_leaderboard,  # pd.DataFrame
        remaining_submissions: int,
        remaining_iterations: int,
    ) -> StrategyAdvice:
        user_message = self._build_prompt(
            meta, history, autogluon_leaderboard, remaining_submissions, remaining_iterations
        )

        response = self.client.messages.create(
            model=self.config.claude_model,
            max_tokens=1024,
            temperature=0.3,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Strategy response parse error: {e}. Returning default advice.")
            return StrategyAdvice(summary="Could not parse strategy response; continuing with defaults.")

        return StrategyAdvice(
            summary=data.get("summary", ""),
            next_preset_override=data.get("next_preset_override"),
            feature_engineering_additions=data.get("feature_engineering_additions", []),
            drop_features=data.get("drop_features", []),
            autogluon_extra_args=data.get("autogluon_extra_args", {}),
            should_stop=bool(data.get("should_stop", False)),
            stop_reason=data.get("stop_reason"),
        )

    @staticmethod
    def _build_prompt(
        meta: CompetitionMeta,
        history: list[IterationSummary],
        leaderboard,
        remaining_submissions: int,
        remaining_iterations: int,
    ) -> str:
        history_lines = []
        for s in history:
            lb = f"{s.public_lb_score:.5f}" if s.public_lb_score is not None else "N/A"
            history_lines.append(
                f"  Iter {s.iteration}: preset={s.preset_used}, "
                f"val={s.val_score:.5f}, public_lb={lb}, "
                f"strategy={s.strategy_applied or 'default'}"
            )

        lb_summary = ""
        try:
            top5 = leaderboard.head(5)[["model", "score_val"]].to_string(index=False)
            lb_summary = f"\nTop AutoGluon models:\n{top5}"
        except Exception:
            pass

        return (
            f"Competition: {meta.title}\n"
            f"Task: {meta.task_type}, Metric: {meta.eval_metric} ({meta.eval_metric_direction})\n"
            f"Feature notes: {', '.join(meta.feature_notes[:5])}\n"
            f"\nScore history ({meta.eval_metric_direction}):\n" + "\n".join(history_lines) +
            lb_summary +
            f"\n\nRemaining Kaggle submissions today: {remaining_submissions}"
            f"\nRemaining iterations planned: {remaining_iterations}"
            f"\n\nWhat should we do next?"
        )
