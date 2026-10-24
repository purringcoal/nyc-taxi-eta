from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import add_time_features

model = joblib.load(Path("models/lgbm.joblib"))

app = FastAPI(title="Taxi ETA")


class TripRequest(BaseModel):
    pickup_datetime: datetime
    PULocationID: int = Field(ge=1, le=265)
    DOLocationID: int = Field(ge=1, le=265)


class TripResponse(BaseModel):
    duration_min: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=TripResponse)
def predict(req: TripRequest) -> TripResponse:
    df = pd.DataFrame([req.model_dump()])
    df = add_time_features(df, "pickup_datetime")
    pred = float(model.predict(df)[0])
    return TripResponse(duration_min=round(pred, 1))