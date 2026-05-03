import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from .competition_reader import CompetitionMeta
from .trainer import TrainingResult
from .submitter import SubmissionRecord

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class IterationSummary:
    iteration: int
    val_score: float
    public_lb_score: Optional[float]
    preset_used: str
    duration_seconds: float
    strategy_applied: Optional[str]
    delta_from_best: Optional[float] = None


class ScoreTracker:
    def __init__(self, meta: CompetitionMeta, workspace: Path):
        self.meta = meta
        self.workspace = workspace
        self.history: list[IterationSummary] = []

    def record(self, training_result: TrainingResult, submission_record: SubmissionRecord) -> IterationSummary:
        summary = IterationSummary(
            iteration=training_result.iteration,
            val_score=training_result.val_score,
            public_lb_score=submission_record.public_lb_score,
            preset_used=training_result.preset_used,
            duration_seconds=training_result.training_duration_seconds,
            strategy_applied=submission_record.strategy_applied,
        )

        best = self.get_best()
        if best is not None:
            score_a = summary.public_lb_score or summary.val_score
            score_b = best.public_lb_score or best.val_score
            if self.meta.eval_metric_direction == "maximize":
                summary.delta_from_best = score_a - score_b
            else:
                summary.delta_from_best = score_b - score_a

        self.history.append(summary)
        return summary

    def get_best(self) -> Optional[IterationSummary]:
        if not self.history:
            return None

        def score_key(s: IterationSummary) -> float:
            v = s.public_lb_score if s.public_lb_score is not None else s.val_score
            return v if self.meta.eval_metric_direction == "maximize" else -v

        return max(self.history, key=score_key)

    def is_improving(self, window: int = 3) -> bool:
        if len(self.history) < 2:
            return True
        recent = self.history[-window:]
        deltas = [s.delta_from_best for s in recent if s.delta_from_best is not None]
        return any(d > 0 for d in deltas)

    def get_history(self) -> list[IterationSummary]:
        return list(self.history)

    def print_summary_table(self) -> None:
        table = Table(title=f"Score History — {self.meta.eval_metric}", show_lines=True)
        table.add_column("Iter", style="cyan", justify="center")
        table.add_column("Preset", style="magenta")
        table.add_column("Val Score", justify="right")
        table.add_column("Public LB", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Time (s)", justify="right")

        for s in self.history:
            delta_str = ""
            if s.delta_from_best is not None:
                color = "green" if s.delta_from_best > 0 else "red"
                sign = "+" if s.delta_from_best > 0 else ""
                delta_str = f"[{color}]{sign}{s.delta_from_best:.4f}[/{color}]"

            lb_str = f"{s.public_lb_score:.5f}" if s.public_lb_score is not None else "pending"
            table.add_row(
                str(s.iteration),
                s.preset_used,
                f"{s.val_score:.5f}",
                lb_str,
                delta_str,
                f"{s.duration_seconds:.0f}",
            )

        console.print(table)
