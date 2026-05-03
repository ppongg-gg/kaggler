import pytest
from pathlib import Path
from unittest.mock import MagicMock
from kaggler.competition_reader import CompetitionMeta
from kaggler.tracker import ScoreTracker, IterationSummary
from kaggler.trainer import TrainingResult
from kaggler.submitter import SubmissionRecord


@pytest.fixture
def meta():
    return CompetitionMeta(
        title="Titanic",
        task_type="binary_classification",
        target_column="Survived",
        eval_metric="accuracy",
        eval_metric_direction="maximize",
        id_column="PassengerId",
    )


def make_result(iteration: int, val_score: float, lb_score=None) -> tuple:
    training = MagicMock(spec=TrainingResult)
    training.iteration = iteration
    training.val_score = val_score
    training.preset_used = "medium_quality"
    training.training_duration_seconds = 100.0

    submission = MagicMock(spec=SubmissionRecord)
    submission.public_lb_score = lb_score
    submission.strategy_applied = None
    return training, submission


def test_get_best_maximize(meta, tmp_path):
    tracker = ScoreTracker(meta, tmp_path)
    tracker.record(*make_result(1, 0.78, 0.75))
    tracker.record(*make_result(2, 0.82, 0.80))
    tracker.record(*make_result(3, 0.80, 0.77))

    best = tracker.get_best()
    assert best.iteration == 2


def test_is_improving_false_when_stagnant(meta, tmp_path):
    tracker = ScoreTracker(meta, tmp_path)
    tracker.record(*make_result(1, 0.80, 0.80))
    tracker.record(*make_result(2, 0.80, 0.80))
    tracker.record(*make_result(3, 0.79, 0.79))

    assert not tracker.is_improving(window=3)


def test_delta_is_positive_on_improvement(meta, tmp_path):
    tracker = ScoreTracker(meta, tmp_path)
    tracker.record(*make_result(1, 0.78, 0.75))
    tracker.record(*make_result(2, 0.82, 0.80))

    history = tracker.get_history()
    assert history[1].delta_from_best > 0
