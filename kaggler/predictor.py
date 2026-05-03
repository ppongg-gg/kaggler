import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .competition_reader import CompetitionMeta

logger = logging.getLogger(__name__)

_PROBA_METRICS = {"roc_auc", "auc", "logloss", "log_loss", "binary_logloss"}


class Predictor:
    def __init__(self, meta: CompetitionMeta):
        self.meta = meta

    def generate(
        self,
        predictor,  # AutoGluon TabularPredictor
        test_df: pd.DataFrame,
        sample_submission: Optional[pd.DataFrame],
        iteration: int,
        workspace: Path,
    ) -> Path:
        submissions_dir = workspace / "submissions"
        submissions_dir.mkdir(parents=True, exist_ok=True)
        output_path = submissions_dir / f"iteration_{iteration}.csv"

        use_proba = self._use_predict_proba()

        if use_proba:
            proba = predictor.predict_proba(test_df)
            # For binary classification take the positive class probability
            if isinstance(proba, pd.DataFrame):
                pos_class = proba.columns[-1]
                predictions = proba[pos_class]
            else:
                predictions = proba
        else:
            predictions = predictor.predict(test_df)

        submission = self._build_submission_df(predictions, test_df, sample_submission)
        submission.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path} ({len(submission)} rows)")
        return output_path

    def _use_predict_proba(self) -> bool:
        metric = (self.meta.eval_metric or "").lower().replace("-", "_")
        task = self.meta.task_type
        return metric in _PROBA_METRICS and task == "binary_classification"

    def _build_submission_df(
        self,
        predictions: pd.Series,
        test_df: pd.DataFrame,
        sample_submission: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        if sample_submission is not None:
            # Align to sample_submission format exactly
            result = sample_submission.copy()
            target_col = self.meta.target_column
            if target_col in result.columns:
                # Reset index to align by position
                result[target_col] = predictions.values
                return result

        # Build from scratch
        df = pd.DataFrame()
        if self.meta.id_column and self.meta.id_column in test_df.columns:
            df[self.meta.id_column] = test_df[self.meta.id_column].values
        df[self.meta.target_column] = predictions.values
        return df
