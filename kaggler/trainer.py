import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .competition_reader import CompetitionMeta

logger = logging.getLogger(__name__)

ITERATION_PRESETS = {
    1: {"presets": "medium_quality", "time_limit": 300},
    2: {"presets": "good_quality",   "time_limit": 900},
    3: {"presets": "high_quality",   "time_limit": 1800},
    4: {"presets": "best_quality",   "time_limit": 3600},
}

# Iterations beyond 4 use best_quality with increasing time limits
_BASE_TIME_BEYOND_4 = 3600
_TIME_INCREMENT = 1800


@dataclass
class TrainingResult:
    iteration: int
    model_path: Path
    best_model_name: str
    val_score: float
    training_duration_seconds: float
    preset_used: str
    leaderboard: object = field(default=None, repr=False)  # pd.DataFrame


class Trainer:
    def __init__(self, meta: CompetitionMeta, workspace: Path):
        self.meta = meta
        self.workspace = workspace

    def train(
        self,
        train_df: pd.DataFrame,
        iteration: int,
        strategy_hints: Optional[dict] = None,
    ) -> TrainingResult:
        from autogluon.tabular import TabularPredictor

        ag_config = self._get_ag_config(iteration, strategy_hints)
        model_path = self.workspace / "models" / f"iteration_{iteration}"
        model_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Training iteration {iteration} with preset='{ag_config['presets']}', "
            f"time_limit={ag_config['time_limit']}s"
        )
        start = time.time()

        predictor = TabularPredictor(
            label=self.meta.target_column,
            eval_metric=self.meta.eval_metric or None,
            path=str(model_path),
            verbosity=1,
        ).fit(
            train_data=train_df,
            presets=ag_config["presets"],
            time_limit=ag_config["time_limit"],
            **(ag_config.get("extra_args", {})),
        )

        duration = time.time() - start
        leaderboard = predictor.leaderboard(silent=True)
        best_model = leaderboard.iloc[0]["model"] if len(leaderboard) > 0 else "unknown"
        val_score = leaderboard.iloc[0]["score_val"] if len(leaderboard) > 0 else 0.0

        logger.info(
            f"Iteration {iteration} complete in {duration:.0f}s. "
            f"Best model: {best_model}, val_score: {val_score:.4f}"
        )

        return TrainingResult(
            iteration=iteration,
            model_path=model_path,
            best_model_name=str(best_model),
            val_score=float(val_score),
            training_duration_seconds=duration,
            preset_used=ag_config["presets"],
            leaderboard=leaderboard,
        )

    def load_predictor(self, iteration: int):
        from autogluon.tabular import TabularPredictor
        model_path = self.workspace / "models" / f"iteration_{iteration}"
        return TabularPredictor.load(str(model_path))

    def _get_ag_config(self, iteration: int, strategy_hints: Optional[dict]) -> dict:
        if iteration <= 4:
            config = dict(ITERATION_PRESETS[iteration])
        else:
            extra_time = (iteration - 4) * _TIME_INCREMENT
            config = {"presets": "best_quality", "time_limit": _BASE_TIME_BEYOND_4 + extra_time}

        extra_args = {}
        if strategy_hints:
            if strategy_hints.get("next_preset_override"):
                config["presets"] = strategy_hints["next_preset_override"]
            if strategy_hints.get("autogluon_extra_args"):
                extra_args.update(strategy_hints["autogluon_extra_args"])

        config["extra_args"] = extra_args
        return config
