import pandas as pd
import pytest
from kaggler.competition_reader import CompetitionMeta
from kaggler.data_manager import DataInventory
from kaggler.preprocessor import Preprocessor
from pathlib import Path


@pytest.fixture
def meta():
    return CompetitionMeta(
        title="Titanic",
        task_type="binary_classification",
        target_column="Survived",
        eval_metric="accuracy",
        eval_metric_direction="maximize",
        id_column="PassengerId",
        feature_notes=["Cabin has many NaNs"],
        suggested_features=[],
    )


@pytest.fixture
def inventory(tmp_path):
    dummy = tmp_path / "train.csv"
    dummy.write_text("PassengerId,Survived,Age,Name\n1,1,22,Mr. Smith\n")
    return DataInventory(
        train_file=dummy,
        test_file=dummy,
        sample_submission_file=None,
        train_shape=(1, 4),
        test_shape=(1, 4),
        column_names=["PassengerId", "Survived", "Age", "Name"],
    )


def test_build_plan_drops_id_column(meta, inventory):
    p = Preprocessor(meta, inventory)
    plan = p.build_plan()
    assert "PassengerId" in plan.drop_columns


def test_fit_transform_removes_id(meta, inventory):
    df = pd.DataFrame({
        "PassengerId": [1, 2],
        "Survived": [1, 0],
        "Age": [22, 35],
        "Name": ["Mr. Smith", "Mrs. Jones"],
    })
    p = Preprocessor(meta, inventory)
    p.build_plan()
    result = p.fit_transform(df)
    assert "PassengerId" not in result.columns
    assert "Survived" in result.columns


def test_transform_does_not_require_target(meta, inventory):
    df_test = pd.DataFrame({
        "PassengerId": [3],
        "Age": [40],
        "Name": ["Miss. White"],
    })
    train_df = pd.DataFrame({
        "PassengerId": [1],
        "Survived": [1],
        "Age": [22],
        "Name": ["Mr. Smith"],
    })
    p = Preprocessor(meta, inventory)
    p.build_plan()
    p.fit_transform(train_df)
    result = p.transform(df_test)
    assert "PassengerId" not in result.columns
