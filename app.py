#!/usr/bin/env python3
"""FastAPI service for term deposit subscription scoring."""

from __future__ import annotations

import joblib
from fastapi import FastAPI, HTTPException
from pathlib import Path

from term_deposit.schemas import PredictionRequest, PredictionResponse

APP = FastAPI(title="Term Deposit Subscription API", version="1.0.0")

MODEL_PATH = Path("artifacts/model.joblib")
PIPELINE = None


def load_pipeline() -> None:
    global PIPELINE
    if MODEL_PATH.exists():
        PIPELINE = joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(
            "Model artifact not found. Train the model first (python train.py)."
        )


@APP.on_event("startup")
async def startup_event() -> None:
    try:
        load_pipeline()
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc


@APP.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    if PIPELINE is None:
        raise HTTPException(status_code=500, detail="Model pipeline not loaded.")

    payload = request.dict()
    df = build_request_frame(payload)
    proba = PIPELINE.predict_proba(df)[0, 1]
    label = "yes" if proba >= 0.5 else "no"
    return PredictionResponse(subscribe_proba=float(proba), subscribe_label=label)


def build_request_frame(payload: dict) -> "pd.DataFrame":
    import pandas as pd

    row = {
        "age": payload["age"],
        "balance": payload["balance"],
        "day": payload["day"],
        "duration": payload["duration"],
        "campaign": payload["campaign"],
        "job": payload["job"],
        "marital": payload["marital"],
        "education": payload["education"],
        "default": payload["default"],
        "housing": payload["housing"],
        "loan": payload["loan"],
        "contact": payload["contact"],
        "month": payload["month"],
    }
    return pd.DataFrame([row])


if __name__ == "__main__":
    import uvicorn

    load_pipeline()
    uvicorn.run(APP, host="0.0.0.0", port=8000)
