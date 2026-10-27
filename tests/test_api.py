import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app, get_model
from src.features import FEATURES


class FakeModel:
    """Подмена модели из реестра. Запоминает вход и возвращает фиксированное число."""

    def __init__(self):
        self.last_input = None

    def predict(self, df):
        self.last_input = df
        return np.array([12.345])


@pytest.fixture
def fake_model():
    model = FakeModel()
    app.dependency_overrides[get_model] = lambda: model
    yield model
    app.dependency_overrides.clear()


@pytest.fixture
def client(fake_model):
    # TestClient без with не запускает lifespan, поэтому модель из MLflow не загружается.
    return TestClient(app)


REQUEST = {"pickup_datetime": "2024-04-01T08:30:00", "PULocationID": 132, "DOLocationID": 230}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_rounded_duration(client):
    response = client.post("/predict", json=REQUEST)
    assert response.status_code == 200
    assert response.json() == {"duration_min": 12.3}


def test_model_gets_features_in_training_format(client, fake_model):
    client.post("/predict", json=REQUEST)
    df = fake_model.last_input
    assert list(df.columns) == FEATURES
    assert all(dtype == "int64" for dtype in df.dtypes)
    assert df.iloc[0].to_dict() == {"PULocationID": 132, "DOLocationID": 230,
                                    "pickup_hour": 8, "pickup_dow": 0, "pickup_month": 4}


@pytest.mark.parametrize("bad", [
    {**REQUEST, "PULocationID": 999},
    {**REQUEST, "DOLocationID": 0},
    {**REQUEST, "pickup_datetime": "вчера"},
    {"PULocationID": 132, "DOLocationID": 230},
])
def test_bad_requests_are_rejected(client, bad):
    assert client.post("/predict", json=bad).status_code == 422