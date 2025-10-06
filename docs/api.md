# API Quickstart

## 1. Train the model

```
PYTHONPATH=src python train.py --data term-deposit-marketing-2020.csv --output-dir artifacts
```

This writes `artifacts/model.joblib` and `artifacts/metrics.json`.

## 2. Start the FastAPI server

```
PYTHONPATH=src uvicorn app:APP --reload --port 8000
```

## 3. Example request

```
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
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
    "month": "may"
  }'
```

Response:

```
{
  "subscribe_proba": 0.78,
  "subscribe_label": "yes"
}
```
