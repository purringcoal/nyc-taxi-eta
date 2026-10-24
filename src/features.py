from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
TARGET = "duration_min"
PAIR = ["PULocationID", "DOLocationID"]
FEATURES = ["PULocationID", "DOLocationID", "pickup_hour", "pickup_dow"]
CAT_FEATURES = ["PULocationID", "DOLocationID"]

def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Одна и та же функция для обучения и для API."""
    df = df.copy()
    t = pd.to_datetime(df[time_col])
    df["pickup_hour"] = t.dt.hour
    df["pickup_dow"] = t.dt.dayofweek #0 - Monday, 6 - Sunday
    return df


def load_month(month: str) -> pd.DataFrame:
    path = RAW_DIR / f"yellow_tripdata_{month}.parquet"
    cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", *PAIR]
    df = pd.read_parquet(path, columns=cols)

    # Оставляем поздки, начавшиеся в этом месяце
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.DateOffset(months=1)
    t = df["tpep_pickup_datetime"]
    df = df[(t >= start) & (t < end)]

    # Целевая переменная и фильтрация выбросов
    dur = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    df = df.assign(**{TARGET: dur})
    df = df[df[TARGET].between(1, 120)]
    return add_time_features(df, "tpep_pickup_datetime")
