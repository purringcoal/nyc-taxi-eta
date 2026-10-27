import pandas as pd
import pytest

from src import validation
from src.features import raw_path
from src.validation import validate_month
from tests.conftest import make_raw_month


@pytest.fixture(autouse=True)
def small_min_rows(monkeypatch):
    # В тестовых файлах тысячи строк, а не миллионы.
    monkeypatch.setattr(validation, "MIN_ROWS", 100)


def test_good_month_passes(write_month):
    write_month("2024-03")
    report = validate_month("2024-03")
    assert report.ok, report.errors
    assert report.n_rows == 2000
    assert report.stats["in_month_share"] == 1.0


def test_missing_file(raw_dir):
    report = validate_month("2024-03")
    assert not report.ok
    assert "не найден" in report.errors[0]


def test_truncated_file(write_month):
    write_month("2024-03")
    path = raw_path("2024-03")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    report = validate_month("2024-03")
    assert not report.ok
    assert "не читается" in report.errors[0]


def test_wrong_zone_type(write_month):
    df = make_raw_month("2024-03")
    write_month("2024-03", df.assign(PULocationID=df["PULocationID"].astype(str)))
    report = validate_month("2024-03")
    assert not report.ok
    assert "PULocationID" in report.errors[0]


def test_too_few_rows(write_month):
    write_month("2024-03", make_raw_month("2024-03", n=50))
    assert not validate_month("2024-03").ok


def test_trips_from_other_month(write_month):
    df = make_raw_month("2024-03")
    df.loc[:199, "tpep_pickup_datetime"] = pd.Timestamp("2002-12-31")
    write_month("2024-03", df)
    report = validate_month("2024-03")
    assert not report.ok
    assert any("in_month_share" in e for e in report.errors)


def test_new_column_is_only_a_warning(write_month):
    write_month("2024-12")
    write_month("2025-01", make_raw_month("2025-01").assign(cbd_congestion_fee=0.75))
    reference = validation.read_columns("2024-12")
    report = validate_month("2025-01", reference)
    assert report.ok
    assert report.warnings == ["новые колонки ['cbd_congestion_fee']"]