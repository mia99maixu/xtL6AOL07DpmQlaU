"""Unit tests for the term deposit pipeline module."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from term_deposit import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TrainConfig,
    build_pipeline,
    prepare_features,
    train_model,
)


@pytest.fixture()
def sample_frame() -> pd.DataFrame:
    """Create a small synthetic dataset covering the required columns."""

    rows = [
        {
            "age": 42,
            "balance": 6000,
            "day": 10,
            "duration": 320,
            "campaign": 1,
            "job": "management",
            "marital": "married",
            "education": "tertiary",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "month": "may",
            "y": "yes",
        },
        {
            "age": 44,
            "balance": 4200,
            "day": 12,
            "duration": 310,
            "campaign": 1,
            "job": "admin.",
            "marital": "married",
            "education": "tertiary",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "month": "may",
            "y": "yes",
        },
        {
            "age": 37,
            "balance": 1200,
            "day": 5,
            "duration": 180,
            "campaign": 2,
            "job": "technician",
            "marital": "single",
            "education": "secondary",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "telephone",
            "month": "apr",
            "y": "no",
        },
        {
            "age": 35,
            "balance": 900,
            "day": 7,
            "duration": 160,
            "campaign": 2,
            "job": "services",
            "marital": "single",
            "education": "secondary",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "telephone",
            "month": "apr",
            "y": "no",
        },
        {
            "age": 50,
            "balance": 2500,
            "day": 15,
            "duration": 240,
            "campaign": 1,
            "job": "blue-collar",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "month": "jun",
            "y": "no",
        },
        {
            "age": 29,
            "balance": 400,
            "day": 20,
            "duration": 100,
            "campaign": 3,
            "job": "services",
            "marital": "single",
            "education": "secondary",
            "default": "no",
            "housing": "yes",
            "loan": "yes",
            "contact": "cellular",
            "month": "aug",
            "y": "yes",
        },
        {
            "age": 31,
            "balance": 700,
            "day": 21,
            "duration": 110,
            "campaign": 3,
            "job": "services",
            "marital": "single",
            "education": "secondary",
            "default": "no",
            "housing": "yes",
            "loan": "yes",
            "contact": "cellular",
            "month": "aug",
            "y": "yes",
        },
    ]
    return pd.DataFrame(rows)


def test_prepare_features_returns_expected_shapes(sample_frame: pd.DataFrame) -> None:
    X, y = prepare_features(sample_frame)
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert set(y.unique()) <= {0, 1}
    assert y.sum() >= 1  # ensure positives exist after mapping


def test_train_model_runs_successfully(sample_frame: pd.DataFrame) -> None:
    result = train_model(sample_frame, TrainConfig(test_size=0.25, cv_splits=2))
    assert set(result.metrics.keys()) >= {
        "accuracy",
        "precision_positive",
        "recall_positive",
        "f1_positive",
        "roc_auc",
    }
    assert set(result.cv_metrics.keys()) == {"accuracy", "f1"}
    assert result.pipeline is not None


def test_missing_columns_raise_error(sample_frame: pd.DataFrame) -> None:
    df = sample_frame.drop(columns=[NUMERIC_FEATURES[0]])
    with pytest.raises(ValueError):
        prepare_features(df)


def test_build_pipeline_structure() -> None:
    pipeline = build_pipeline()
    assert pipeline.named_steps["preprocessor"] is not None
    assert pipeline.named_steps["model"] is not None
