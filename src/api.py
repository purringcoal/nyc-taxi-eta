import os
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache

import mlflow
import pandas as pd
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from src.features import MAX_ZONE, MIN_ZONE, add_time_features, make_model_input

MODEL_URI = os.environ.get("MODEL_URI", "models:/taxi-eta@champion")


@lru_cache(maxsize=1)
def get_model():
    """Загружает модель из реестра MLflow. Повторные вызовы возвращают уже загруженную."""
    return mlflow.pyfunc.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Загружаем модель при старте сервиса, чтобы ошибка загрузки была видна сразу,
    # а не на первом запросе.
    get_model()
    yield


app = FastAPI(title="Taxi ETA", lifespan=lifespan)


class TripRequest(BaseModel):
    pickup_datetime: datetime
    PULocationID: int = Field(ge=MIN_ZONE, le=MAX_ZONE)
    DOLocationID: int = Field(ge=MIN_ZONE, le=MAX_ZONE)


class TripResponse(BaseModel):
    duration_min: float


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict", response_model=TripResponse)
def predict(req: TripRequest, model=Depends(get_model)) -> TripResponse:
    df = pd.DataFrame([req.model_dump()])
    df = add_time_features(df, "pickup_datetime")
    pred = float(model.predict(make_model_input(df))[0])
    return TripResponse(duration_min=round(pred, 1))