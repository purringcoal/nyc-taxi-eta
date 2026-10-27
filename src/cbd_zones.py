"""Определение зон TLC, которые лежат внутри зоны платы за въезд.

С 5 января 2025 года в данных есть колонка cbd_congestion_fee. Это сбор, который
берётся с поездки, если она начинается, заканчивается или проходит через зону
платы. Значит, поездки, начавшиеся внутри зоны платы, платят сбор всегда.
Зону TLC считаем лежащей внутри зоны платы, если почти все поездки из неё
платят сбор.
"""
import json
from pathlib import Path

import pandas as pd

from src.download import ZONE_LOOKUP_PATH, download_zone_lookup
from src.features import raw_path

# Февраль 2025 года целиком приходится на время после введения платы.
# В январе первые четыре дня сбора ещё не было, и доли получились бы ниже.
REFERENCE_MONTH = "2025-02"
# Доли у зон внутри зоны платы лежат от 0.94 до 0.99, у ближайших зон снаружи
# не выше 0.58. Порог выбран посередине этого разрыва.
MIN_FEE_SHARE = 0.75
MIN_TRIPS = 1000  # в зонах с малым числом поездок доля слишком шумная
CBD_ZONES_PATH = Path("configs/cbd_zones.json")


def fee_share_by_zone(month: str) -> pd.DataFrame:
    """Для каждой зоны посадки доля поездок, которые заплатили сбор, и число поездок."""
    df = pd.read_parquet(raw_path(month), columns=["PULocationID", "cbd_congestion_fee"])
    paid = df["cbd_congestion_fee"].fillna(0) > 0
    stats = paid.groupby(df["PULocationID"]).agg(fee_share="mean", trips="size")
    return stats.rename_axis("LocationID").reset_index()


def find_cbd_zones(stats: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Отбирает зоны Манхэттена, где почти все поездки платят сбор."""
    stats = stats.merge(lookup, on="LocationID", how="left")
    is_cbd = (
        (stats["Borough"] == "Manhattan")
        & (stats["trips"] >= MIN_TRIPS)
        & (stats["fee_share"] >= MIN_FEE_SHARE)
    )
    return stats[is_cbd].sort_values("LocationID")


def load_cbd_zones() -> set[int]:
    """Множество номеров зон внутри зоны платы, сохранённое скриптом этого модуля."""
    data = json.loads(CBD_ZONES_PATH.read_text(encoding="utf-8"))
    return set(data["zones"])


def main():
    download_zone_lookup()
    lookup = pd.read_csv(ZONE_LOOKUP_PATH)
    stats = fee_share_by_zone(REFERENCE_MONTH)
    cbd = find_cbd_zones(stats, lookup)

    # Пограничные зоны Манхэттена показываем отдельно, чтобы их можно было
    # проверить глазами. Это зоны, где сбор платит заметная, но не подавляющая доля поездок.
    manhattan = stats.merge(lookup, on="LocationID").query("Borough == 'Manhattan' and trips >= @MIN_TRIPS")
    borderline = manhattan[manhattan["fee_share"].between(0.5, MIN_FEE_SHARE, inclusive="left")]

    columns = ["LocationID", "Zone", "fee_share", "trips"]
    print(f"Зоны внутри зоны платы ({len(cbd)} шт.)")
    print(cbd[columns].to_string(index=False))
    print()
    print("Пограничные зоны, не вошедшие в список")
    print(borderline.sort_values("fee_share", ascending=False)[columns].to_string(index=False))

    CBD_ZONES_PATH.parent.mkdir(exist_ok=True)
    data = {
        "reference_month": REFERENCE_MONTH,
        "min_fee_share": MIN_FEE_SHARE,
        "zones": cbd["LocationID"].astype(int).tolist(),
    }
    CBD_ZONES_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nСписок сохранён в {CBD_ZONES_PATH}")


if __name__ == "__main__":
    main()