"""Smoke tests for the FastAPI prediction endpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import APP, load_pipeline, MODEL_PATH  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ensure_model(tmp_path_factory):
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not available; run train.py first.")
    load_pipeline()


def test_predict_endpoint():
    client = TestClient(APP)
    payload = {
        "age": 42,
        "balance": 6000.0,
        "day": 10,
        "duration": 320.0,
        "campaign": 1,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "housing": "no",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "subscribe_proba" in body
    assert 0.0 <= body["subscribe_proba"] <= 1.0
    assert body["subscribe_label"] in {"yes", "no"}
