from pathlib import Path

import pandas as pd

from src.months import month_bounds

RAW_DIR = Path("data/raw")

TARGET = "duration_min"
PAIR = ["PULocationID", "DOLocationID"]
FEATURES = ["PULocationID", "DOLocationID", "pickup_hour", "pickup_dow", "pickup_month"]
CAT_FEATURES = ["PULocationID", "DOLocationID"]

# Колонки, которые мы читаем из исходного файла. Остальные нам не нужны.
RAW_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]

# Правила очистки. Ими пользуются и очистка, и валидация, чтобы правила не расходились.
MIN_DURATION, MAX_DURATION = 1, 120  # минуты
MIN_ZONE, MAX_ZONE = 1, 265  # номера зон TLC


def raw_path(month: str) -> Path:
    return RAW_DIR / f"yellow_tripdata_{month}.parquet"


def read_raw_month(month: str) -> pd.DataFrame:
    """Читает из файла месяца только нужные колонки, без какой-либо очистки."""
    return pd.read_parquet(raw_path(month), columns=RAW_COLUMNS)


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет целевую переменную, длительность поездки в минутах."""
    seconds = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
    return df.assign(**{TARGET: seconds / 60})


def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Одна и та же функция для обучения и для API."""
    df = df.copy()
    t = pd.to_datetime(df[time_col])
    df["pickup_hour"] = t.dt.hour
    df["pickup_dow"] = t.dt.day_of_week  # 0 соответствует понедельнику
    df["pickup_month"] = t.dt.month
    return df


def clean_month(df: pd.DataFrame, month: str) -> pd.DataFrame:
    """Очищает сырые данные месяца и добавляет целевую переменную и признаки."""
    # Строки с пропусками в нужных колонках использовать нельзя.
    df = df.dropna(subset=RAW_COLUMNS)

    # Оставляем поездки, которые начались внутри этого месяца.
    start, end = month_bounds(month)
    t = df["tpep_pickup_datetime"]
    df = df[(t >= start) & (t < end)]

    # Убираем слишком короткие и слишком длинные поездки.
    df = add_duration(df)
    df = df[df[TARGET].between(MIN_DURATION, MAX_DURATION)]

    # Убираем несуществующие номера зон. API такие запросы тоже не пропускает.
    zones_ok = df["PULocationID"].between(MIN_ZONE, MAX_ZONE) & df["DOLocationID"].between(MIN_ZONE, MAX_ZONE)
    df = df[zones_ok]

    # Если в колонке были пропуски, pandas мог прочитать её как float.
    # После dropna приводим зоны обратно к целым числам.
    df = df.astype({col: "int64" for col in PAIR})

    return add_time_features(df, "tpep_pickup_datetime")


def load_month(month: str) -> pd.DataFrame:
    """Читает и очищает месяц. Это основная функция для обучения."""
    return clean_month(read_raw_month(month), month)

def load_sample(months: list[str], n_per_month: int, seed: int = 42) -> pd.DataFrame:
    """Случайная подвыборка из нескольких месяцев, поровну из каждого.

    Поровну, чтобы каждый сезон был представлен одинаково, даже если
    в каких-то месяцах поездок больше.
    """
    parts = []
    for month in months:
        df = load_month(month)
        parts.append(df.sample(n=min(n_per_month, len(df)), random_state=seed))
    return pd.concat(parts, ignore_index=True) 


def make_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Готовит вход модели. Оставляет только признаки в фиксированном порядке и приводит их к int64."""
    return df[FEATURES].astype("int64")