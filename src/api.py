import os
from datetime import datetime

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import add_time_features, make_model_input

MODEL_URI = os.environ.get("MODEL_URI", "models:/taxi-eta@champion")
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(title="Taxi ETA")


class TripRequest(BaseModel):
    pickup_datetime: datetime
    PULocationID: int = Field(ge=1, le=265)
    DOLocationID: int = Field(ge=1, le=265)


class TripResponse(BaseModel):
    duration_min: float


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict", response_model=TripResponse)
def predict(req: TripRequest) -> TripResponse:
    df = pd.DataFrame([req.model_dump()])
    df = add_time_features(df, "pickup_datetime")
    pred = float(model.predict(make_model_input(df))[0])
    return TripResponse(duration_min=round(pred, 1))