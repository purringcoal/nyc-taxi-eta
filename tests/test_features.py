import pandas as pd

from src.features import FEATURES, TARGET, add_time_features, clean_month, load_sample, make_model_input
from tests.conftest import make_raw_month


def trip(trip_id, pickup, minutes, pu=100, do=200):
    pickup = pd.Timestamp(pickup)
    return {"trip_id": trip_id, "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup + pd.Timedelta(minutes=minutes),
            "PULocationID": pu, "DOLocationID": do}


def test_clean_month_applies_every_rule():
    raw = pd.DataFrame([
        trip("обычная", "2024-03-10 08:00", 10),
        trip("ровно 1 минута", "2024-03-10 08:00", 1),
        trip("полминуты", "2024-03-10 08:00", 0.5),
        trip("ровно 120 минут", "2024-03-10 08:00", 120),
        trip("121 минута", "2024-03-10 08:00", 121),
        trip("предыдущий месяц", "2024-02-29 23:59", 10),
        trip("начало следующего месяца", "2024-04-01 00:00", 10),
        trip("зона 0", "2024-03-10 08:00", 10, pu=0),
        trip("зона 266", "2024-03-10 08:00", 10, do=266),
        trip("пропуск зоны", "2024-03-10 08:00", 10, pu=None),
    ])
    clean = clean_month(raw, "2024-03")
    assert set(clean["trip_id"]) == {"обычная", "ровно 1 минута", "ровно 120 минут"}


def test_clean_month_returns_integer_zones_after_nulls():
    # С пропуском pandas делает колонку float, после очистки зоны снова целые.
    raw = pd.DataFrame([trip("a", "2024-03-10 08:00", 10), trip("b", "2024-03-10 09:00", 10, pu=None)])
    assert raw["PULocationID"].dtype == "float64"
    clean = clean_month(raw, "2024-03")
    assert clean["PULocationID"].dtype == "int64"


def test_clean_month_computes_target_in_minutes():
    clean = clean_month(pd.DataFrame([trip("a", "2024-03-10 08:00", 12.5)]), "2024-03")
    assert clean[TARGET].iloc[0] == 12.5


def test_time_features():
    df = add_time_features(pd.DataFrame({"t": [pd.Timestamp("2024-04-01 08:30")]}), "t")
    assert df["pickup_hour"].iloc[0] == 8
    assert df["pickup_dow"].iloc[0] == 0  # 1 апреля 2024 года был понедельник
    assert df["pickup_month"].iloc[0] == 4


def test_model_input_has_fixed_columns_and_int64():
    # Модель в реестре проверяет вход по сигнатуре, и int32 вместо int64 ломает предсказание.
    raw = make_raw_month("2024-03", n=100)
    df = make_model_input(clean_month(raw, "2024-03").assign(extra=1))
    assert list(df.columns) == FEATURES
    assert all(dtype == "int64" for dtype in df.dtypes)


def test_load_sample_takes_equal_share_from_each_month(write_month):
    write_month("2024-01", make_raw_month("2024-01", n=500))
    write_month("2024-02", make_raw_month("2024-02", n=300))
    sample = load_sample(["2024-01", "2024-02"], n_per_month=200)
    assert sample["pickup_month"].value_counts().to_dict() == {1: 200, 2: 200}


def test_load_sample_is_reproducible(write_month):
    write_month("2024-01")
    first = load_sample(["2024-01"], n_per_month=100, seed=7)
    second = load_sample(["2024-01"], n_per_month=100, seed=7)
    pd.testing.assert_frame_equal(first, second)