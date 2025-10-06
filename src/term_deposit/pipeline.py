"""Reusable data preparation and modelling utilities for the term deposit project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------------------
# Feature configuration
# --------------------------------------------------------------------------------------

TARGET = "y"
LABEL_MAP = {"yes": 1, "no": 0}

NUMERIC_FEATURES = ["age", "balance", "day", "duration", "campaign"]
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
]


@dataclass
class TrainConfig:
    """Configuration for training and validation."""

    test_size: float = 0.2
    random_state: int = 42
    cv_splits: int = 5
    lgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "random_state": 42,
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "class_weight": "balanced",
        }
    )


@dataclass
class TrainingResult:
    """Container for training artefacts and evaluation metrics."""

    pipeline: Pipeline
    config: TrainConfig
    metrics: Dict[str, float]
    classification_report: Dict[str, Any]
    cv_metrics: Dict[str, Dict[str, float]]
    X_test: pd.DataFrame
    y_test: pd.Series


# --------------------------------------------------------------------------------------
# Core builders
# --------------------------------------------------------------------------------------


def build_preprocessor() -> ColumnTransformer:
    """Create the column transformer that standardises numeric features and one-hot encodes categoricals."""

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_model(config: TrainConfig | None = None) -> LGBMClassifier:
    """Instantiate the LightGBM classifier using the provided configuration."""

    cfg = config or TrainConfig()
    return LGBMClassifier(**cfg.lgbm_params)


def build_pipeline(config: TrainConfig | None = None) -> Pipeline:
    """Create the full preprocessing + model pipeline."""

    cfg = config or TrainConfig()
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", build_model(cfg)),
        ]
    )


# --------------------------------------------------------------------------------------
# Data preparation helpers
# --------------------------------------------------------------------------------------


def prepare_features(df: pd.DataFrame, target: str = TARGET) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a raw dataframe into feature matrix X and binary target y."""

    missing = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [target]) - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Input dataframe is missing required columns: {missing_cols}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[target].map(LABEL_MAP)

    if y.isna().any():
        raise ValueError("Target column contains unexpected labels; expected 'yes'/'no'.")

    return X, y.astype(int)


# --------------------------------------------------------------------------------------
# Training & evaluation
# --------------------------------------------------------------------------------------


def train_model(df: pd.DataFrame, config: TrainConfig | None = None) -> TrainingResult:
    """Train the LightGBM pipeline and return the fitted artefacts and evaluation metrics."""

    cfg = config or TrainConfig()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    pipeline = build_pipeline(cfg)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, digits=3)
    metrics = {
        "accuracy": float(report["accuracy"]),
        "precision_positive": float(report["1"]["precision"]),
        "recall_positive": float(report["1"]["recall"]),
        "f1_positive": float(report["1"]["f1-score"]),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)

    cv_accuracy = cross_val_score(build_pipeline(cfg), X, y, cv=cv, scoring="accuracy", n_jobs=1)
    cv_f1 = cross_val_score(build_pipeline(cfg), X, y, cv=cv, scoring="f1", n_jobs=1)

    cv_metrics = {
        "accuracy": {"mean": float(np.mean(cv_accuracy)), "std": float(np.std(cv_accuracy))},
        "f1": {"mean": float(np.mean(cv_f1)), "std": float(np.std(cv_f1))},
    }

    return TrainingResult(
        pipeline=pipeline,
        config=cfg,
        metrics=metrics,
        classification_report=report,
        cv_metrics=cv_metrics,
        X_test=X_test,
        y_test=y_test,
    )


__all__ = [
    "TrainConfig",
    "TrainingResult",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "prepare_features",
    "build_preprocessor",
    "build_model",
    "build_pipeline",
    "train_model",
]
