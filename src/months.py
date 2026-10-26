"""Работа с месяцами в формате 'ГГГГ-ММ'."""
import argparse
from datetime import datetime

import pandas as pd


def parse_month(text: str) -> str:
    """Проверяет, что строка похожа на месяц 'ГГГГ-ММ', и возвращает её в стандартном виде."""
    try:
        return datetime.strptime(text, "%Y-%m").strftime("%Y-%m")
    except ValueError:
        raise argparse.ArgumentTypeError(f"ожидался месяц в формате ГГГГ-ММ, получено {text!r}") from None


def month_range(start: str, end: str) -> list[str]:
    """Все месяцы от start до end включительно, например ['2024-11', '2024-12', '2025-01']."""
    periods = pd.period_range(start=start, end=end, freq="M")
    return [p.strftime("%Y-%m") for p in periods]


def month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Начало месяца и начало следующего месяца.

    Правая граница не входит в месяц, поэтому проверка выглядит как start <= t < end.
    """
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.DateOffset(months=1)
    return start, end


def parse_month_range_args(description: str) -> list[str]:
    """Разбирает аргументы командной строки 'начало [конец]' и возвращает список месяцев."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("start", type=parse_month, help="первый месяц, например 2024-01")
    parser.add_argument("end", type=parse_month, nargs="?", help="последний месяц, по умолчанию равен первому")
    args = parser.parse_args()

    end = args.end or args.start
    # Строки вида 'ГГГГ-ММ' сравниваются как даты, потому что у них одинаковая длина
    # и старшие разряды стоят слева.
    if end < args.start:
        parser.error(f"последний месяц {end} раньше первого {args.start}")
    return month_range(args.start, end)