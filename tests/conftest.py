"""Общие заготовки для тестов.

Тесты не читают настоящие данные и не обращаются к серверу MLflow. Все данные
генерируются здесь, маленькими и с заранее известными свойствами.
"""
import numpy as np
import pandas as pd
import pytest

from src import features


def make_raw_month(month: str, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Сырые данные месяца в том же виде, что в файлах TLC, только маленькие и чистые."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(f"{month}-01")
    days = (start + pd.DateOffset(months=1) - start).days
    pickup = start + pd.to_timedelta(rng.integers(0, days * 86400, n), unit="s")
    duration = pd.to_timedelta(rng.integers(2 * 60, 60 * 60, n), unit="s")
    return pd.DataFrame({
        "VendorID": np.full(n, 2, dtype="int32"),
        "tpep_pickup_datetime": pickup.astype("datetime64[us]"),
        "tpep_dropoff_datetime": (pickup + duration).astype("datetime64[us]"),
        "PULocationID": rng.integers(1, 266, n).astype("int32"),
        "DOLocationID": rng.integers(1, 266, n).astype("int32"),
        "fare_amount": rng.uniform(5, 50, n),
    })


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """Временная папка вместо data/raw. После теста pytest удаляет её сам."""
    monkeypatch.setattr(features, "RAW_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def write_month(raw_dir):
    """Функция, которая записывает месяц в parquet во временную папку."""
    def write(month: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
        df = make_raw_month(month) if df is None else df
        df.to_parquet(features.raw_path(month))
        return df
    return write