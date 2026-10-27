"""Проверка сырых данных месяца перед тем, как их использовать.

Проверки делятся на два уровня. Ошибка означает, что данным нельзя доверять,
и пайплайн должен остановиться. Предупреждение означает, что что-то изменилось,
но работать дальше можно. Человеку стоит на это посмотреть.
"""
from dataclasses import dataclass, field

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features import (
    MAX_DURATION,
    MAX_ZONE,
    MIN_DURATION,
    MIN_ZONE,
    PAIR,
    RAW_COLUMNS,
    TARGET,
    add_duration,
    raw_path,
    read_raw_month,
)
from src.months import month_bounds, parse_month_range_args

# Пороги выставлены по истории за 2024-01 .. 2025-06 с запасом относительно
# худшего нормального месяца.
MIN_ROWS = 2_000_000
MAX_NULL_SHARE = 0.001
MIN_IN_MONTH_SHARE = 0.999
MIN_VALID_ZONE_SHARE = 0.999
MIN_VALID_DURATION_SHARE = 0.95

DATETIME_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]


@dataclass
class ValidationReport:
    """Результат проверки одного месяца."""

    month: str
    n_rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def read_columns(month: str) -> set[str]:
    """Имена колонок файла. Читается только схема, сами данные не загружаются."""
    return set(pq.read_schema(raw_path(month)).names)


def check_schema(report: ValidationReport, schema: pa.Schema, reference_columns: set[str] | None) -> None:
    """Проверяет наличие и типы нужных колонок, сравнивает набор колонок с эталонным."""
    columns = set(schema.names)

    missing = [col for col in RAW_COLUMNS if col not in columns]
    if missing:
        report.errors.append(f"нет обязательных колонок {missing}")
        return

    # Типы смотрим по схеме parquet, а не по pandas. Pandas превращает целую колонку
    # с пропусками в float, и по его типам проверка была бы ненадёжной.
    for col in DATETIME_COLUMNS:
        if not pa.types.is_timestamp(schema.field(col).type):
            report.errors.append(f"колонка {col} имеет тип {schema.field(col).type}, ожидалась дата и время")
    for col in PAIR:
        if not pa.types.is_integer(schema.field(col).type):
            report.errors.append(f"колонка {col} имеет тип {schema.field(col).type}, ожидалось целое число")

    # Новые или пропавшие колонки сами по себе не мешают, но о них нужно знать.
    if reference_columns is not None:
        new = sorted(columns - reference_columns)
        gone = sorted(reference_columns - columns)
        if new:
            report.warnings.append(f"новые колонки {new}")
        if gone:
            report.warnings.append(f"пропали колонки {gone}")


def check_share(report: ValidationReport, name: str, value: float, minimum: float) -> None:
    """Сохраняет долю в статистику и добавляет ошибку, если она ниже порога."""
    report.stats[name] = value
    if value < minimum:
        report.errors.append(f"{name} = {value:.4f}, это ниже порога {minimum}")


def check_values(report: ValidationReport, df: pd.DataFrame) -> None:
    """Проверяет объём данных, пропуски и долю строк, проходящих правила очистки."""
    report.n_rows = len(df)
    if report.n_rows < MIN_ROWS:
        report.errors.append(f"всего {report.n_rows:,} строк, ожидалось не меньше {MIN_ROWS:,}")
        return

    # Пропуски считаем по каждой нужной колонке отдельно.
    null_share = df[RAW_COLUMNS].isna().mean()
    report.stats["max_null_share"] = float(null_share.max())
    for col, share in null_share.items():
        if share > MAX_NULL_SHARE:
            report.errors.append(f"в колонке {col} пропущено {share:.2%} значений")

    # Дальше считаем доли на строках без пропусков.
    df = df.dropna(subset=RAW_COLUMNS)

    start, end = month_bounds(report.month)
    t = df["tpep_pickup_datetime"]
    check_share(report, "in_month_share", float(((t >= start) & (t < end)).mean()), MIN_IN_MONTH_SHARE)

    zones_ok = df["PULocationID"].between(MIN_ZONE, MAX_ZONE) & df["DOLocationID"].between(MIN_ZONE, MAX_ZONE)
    check_share(report, "valid_zone_share", float(zones_ok.mean()), MIN_VALID_ZONE_SHARE)

    duration = add_duration(df)[TARGET]
    duration_ok = duration.between(MIN_DURATION, MAX_DURATION)
    check_share(report, "valid_duration_share", float(duration_ok.mean()), MIN_VALID_DURATION_SHARE)

    # Медиана длительности не проверяется порогом. Она нужна, чтобы глазами видеть сдвиги.
    report.stats["median_duration"] = float(duration[duration_ok].median())


def validate_month(month: str, reference_columns: set[str] | None = None) -> ValidationReport:
    """Полная проверка месяца. Если находится ошибка, следующие проверки не выполняются."""
    report = ValidationReport(month)

    path = raw_path(month)
    if not path.exists():
        report.errors.append(f"файл {path} не найден")
        return report

    # Недокачанный или повреждённый parquet обычно не даёт прочитать даже схему.
    try:
        schema = pq.read_schema(path)
    except Exception as exc:
        report.errors.append(f"файл не читается ({exc})")
        return report

    check_schema(report, schema, reference_columns)
    if not report.ok:
        return report

    check_values(report, read_raw_month(month))
    return report


def main():
    months = parse_month_range_args("Проверка сырых данных по месяцам")

    # Эталоном служит предыдущий месяц, прошедший проверку. Так предупреждение
    # появляется один раз, в том месяце, где состав колонок изменился.
    reference_columns = None

    rows = []
    for month in months:
        report = validate_month(month, reference_columns)
        if report.ok:
            reference_columns = read_columns(month)

        rows.append({"month": month, "status": "OK" if report.ok else "ОШИБКА",
                     "rows": report.n_rows, **report.stats})
        for message in report.errors:
            print(f"[{month}] ОШИБКА: {message}")
        for message in report.warnings:
            print(f"[{month}] предупреждение: {message}")

    print()
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

if __name__ == "__main__":
    main()