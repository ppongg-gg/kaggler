import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .competition_reader import CompetitionMeta
from .data_manager import DataInventory

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingPlan:
    target_column: str
    id_column: Optional[str]
    drop_columns: list = field(default_factory=list)
    datetime_columns: list = field(default_factory=list)
    feature_engineering_steps: list = field(default_factory=list)


class Preprocessor:
    """
    Intentionally thin — AutoGluon handles encoding, imputation, and scaling
    internally. This layer only handles structural issues AutoGluon cannot:
    dropping ID columns, datetime parsing, and Claude-suggested custom features.
    """

    def __init__(self, meta: CompetitionMeta, inventory: DataInventory):
        self.meta = meta
        self.inventory = inventory
        self.plan: Optional[PreprocessingPlan] = None
        self._datetime_formats: dict = {}

    def build_plan(self, extra_drops: list = None, extra_features: list = None) -> PreprocessingPlan:
        all_cols = self.inventory.column_names

        drop = []
        if self.meta.id_column and self.meta.id_column in all_cols:
            drop.append(self.meta.id_column)

        # Drop any high-cardinality ID-like columns not caught by meta
        for col in all_cols:
            col_lower = col.lower()
            if col_lower in ("id", "passengerid", "customerid", "userid", "rowid") and col not in drop:
                drop.append(col)

        if extra_drops:
            drop.extend(c for c in extra_drops if c in all_cols and c not in drop)

        # Detect datetime columns from column names
        datetime_hints = ("date", "time", "year", "month", "day", "timestamp")
        datetime_cols = [
            c for c in all_cols
            if any(h in c.lower() for h in datetime_hints)
            and c != self.meta.target_column
            and c not in drop
        ]

        feature_steps = list(self.meta.suggested_features or [])
        if extra_features:
            feature_steps.extend(extra_features)

        self.plan = PreprocessingPlan(
            target_column=self.meta.target_column,
            id_column=self.meta.id_column,
            drop_columns=drop,
            datetime_columns=datetime_cols,
            feature_engineering_steps=feature_steps,
        )
        return self.plan

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.plan is None:
            raise RuntimeError("Call build_plan() before fit_transform()")
        df = df.copy()
        df = self._apply_datetime(df, fit=True)
        df = self._drop_columns(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.plan is None:
            raise RuntimeError("Call build_plan() before transform()")
        df = df.copy()
        df = self._apply_datetime(df, fit=False)
        # Drop id/extra cols but not target (not present in test)
        drop_in_test = [c for c in self.plan.drop_columns if c in df.columns]
        df = df.drop(columns=drop_in_test)
        return df

    def _apply_datetime(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in self.plan.datetime_columns:
            if col not in df.columns:
                continue
            try:
                if fit:
                    parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                else:
                    parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")

                if parsed.notna().sum() > 0:
                    df[f"{col}_year"] = parsed.dt.year
                    df[f"{col}_month"] = parsed.dt.month
                    df[f"{col}_day"] = parsed.dt.day
                    df[f"{col}_dayofweek"] = parsed.dt.dayofweek
                    df = df.drop(columns=[col])
                    logger.debug(f"Expanded datetime column '{col}' into year/month/day/dayofweek")
            except Exception as e:
                logger.warning(f"Could not parse datetime column '{col}': {e}")
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        to_drop = [c for c in self.plan.drop_columns if c in df.columns]
        if to_drop:
            logger.debug(f"Dropping columns: {to_drop}")
            df = df.drop(columns=to_drop)
        return df

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)
