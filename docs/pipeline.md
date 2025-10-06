# Term Deposit Pipeline Docs

## Overview

`term_deposit.pipeline` contains reusable utilities that transform the raw CSV data, train a LightGBM model, and evaluate results. The pipeline mirrors the notebook workflow—scaling numeric fields, one-hot encoding categoricals, and applying a class-weighted LightGBM classifier to focus on the minority "yes" class.

## Key API

- `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES`: lists of feature names used by the pipeline.
- `TrainConfig`: dataclass holding split size, random seed, CV folds, and LightGBM parameters.
- `TrainingResult`: dataclass with the fitted pipeline, metrics, and evaluation artefacts.
- `prepare_features(df)`: converts the raw dataframe into `(X, y)` with binary targets.
- `build_preprocessor()`, `build_model()`, `build_pipeline()`: helpers to compose the repeatable modelling stack.
- `train_model(df, config)`: high-level entry point to train, evaluate, and return a `TrainingResult`.

## CLI Usage

`train.py` offers a simple command-line interface:

```bash
python train.py --data term-deposit-marketing-2020.csv --output-dir artifacts
```

Options:

- `--config config.json`: optional JSON file mirroring `TrainConfig` fields to override defaults.
- `--output-dir`: directory for the serialized pipeline (`model.joblib`) and metrics report (`metrics.json`).

Example `config.json`:

```json
{
  "test_size": 0.3,
  "cv_splits": 3,
  "lgbm_params": {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 50,
    "min_child_samples": 40,
    "class_weight": "balanced",
    "random_state": 123
  }
}
```

Run result summary (printed to stdout) mirrors the notebook metrics—accuracy, minority-class precision/recall/F1, ROC-AUC, plus cross-validation means/std.

## Testing

The `tests/test_pipeline.py` module exercises the key functions using a small synthetic dataset to ensure the pipeline fits, produces expected metrics keys, and reacts appropriately to missing columns.
