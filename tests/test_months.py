import argparse

import pandas as pd
import pytest

from src.months import month_bounds, month_range, months_before, parse_month


def test_month_range_crosses_new_year():
    assert month_range("2024-11", "2025-02") == ["2024-11", "2024-12", "2025-01", "2025-02"]


def test_month_bounds_december():
    start, end = month_bounds("2024-12")
    assert start == pd.Timestamp("2024-12-01")
    assert end == pd.Timestamp("2025-01-01")


def test_months_before_does_not_include_month_itself():
    window = months_before("2025-01", 12)
    assert len(window) == 12
    assert window[0] == "2024-01"
    assert window[-1] == "2024-12"


def test_parse_month_adds_leading_zero():
    assert parse_month("2024-1") == "2024-01"


@pytest.mark.parametrize("text", ["2024-13", "2024/01", "24-01", "2024-01-15", ""])
def test_parse_month_rejects_bad_input(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_month(text)