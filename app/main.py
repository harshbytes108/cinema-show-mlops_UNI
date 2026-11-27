import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.mp183.logger import get_json_logger, log_prediction

FEATURE_COLUMNS = (
    "ticket_price_rs",
    "days_since_release",
    "star_rating",
    "show_hour",
    "screen_seats",
    "distance_to_metro_km",
    "language",
    "location",
)

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_price_rs: float
    days_since_release: int = Field(le=70)
    star_rating: float
    show_hour: int
    screen_seats: int = Field(ge=40)
    distance_to_metro_km: float
    language: str
    location: str

class PredictionResponse(BaseModel):
    prediction: float

@lru_cache(maxsize=1)
def load_model() -> Any:
    model_path = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
    if not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Prediction model is unavailable: {model_path}",
        )
    return joblib.load(model_path)


def get_model() -> Any:
    return load_model()

logger = get_json_logger("mp183_api")
app = FastAPI(title="MP183 Cinema Occupancy API", version="1.0.0")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    model: Any = Depends(get_model),
) -> PredictionResponse:
    start = time.perf_counter()
    payload = request.model_dump()
    features = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    predictions = model.predict(features)
    if len(predictions) != 1:
        raise HTTPException(
            status_code=500,
            detail="Prediction model returned an unexpected number of results",
        )
    pred_value = float(predictions[0])
    latency_ms = (time.perf_counter() - start) * 1000.0
    log_prediction(logger, payload, pred_value, latency_ms)
    return PredictionResponse(prediction=pred_value)
