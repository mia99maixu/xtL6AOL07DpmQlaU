#!/usr/bin/env python3
"""CLI entry point to train and evaluate the term deposit model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from term_deposit import TrainConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the term deposit subscription model")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("term-deposit-marketing-2020.csv"),
        help="Path to the CSV dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store trained model artefacts",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON file overriding TrainConfig parameters",
    )
    return parser.parse_args()


def load_config(config_path: Path | None) -> TrainConfig:
    if not config_path:
        return TrainConfig()

    with config_path.open() as fh:
        overrides = json.load(fh)

    base = TrainConfig()
    params = overrides.get("lgbm_params", base.lgbm_params)

    return TrainConfig(
        test_size=overrides.get("test_size", base.test_size),
        random_state=overrides.get("random_state", base.random_state),
        cv_splits=overrides.get("cv_splits", base.cv_splits),
        lgbm_params=params,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    df = pd.read_csv(args.data)
    result = train_model(df, cfg)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the pipeline and metrics
    import joblib

    pipeline_path = output_dir / "model.joblib"
    joblib.dump(result.pipeline, pipeline_path)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as fh:
        json.dump(
            {
                "metrics": result.metrics,
                "cv_metrics": result.cv_metrics,
                "classification_report": result.classification_report,
                "train_config": {
                    "test_size": result.config.test_size,
                    "random_state": result.config.random_state,
                    "cv_splits": result.config.cv_splits,
                    "lgbm_params": result.config.lgbm_params,
                },
            },
            fh,
            indent=2,
        )

    print(f"Model saved to {pipeline_path}")
    print(f"Metrics saved to {metrics_path}")
    print("Summary metrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.3f}")
    print("CV metrics:")
    for metric, stats in result.cv_metrics.items():
        print(f"  {metric}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")


if __name__ == "__main__":
    main()
