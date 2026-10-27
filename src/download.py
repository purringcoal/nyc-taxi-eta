"""Загрузка данных жёлтого такси за диапазон месяцев и справочника зон."""
from pathlib import Path
from urllib.request import urlretrieve

from src.features import RAW_DIR, raw_path
from src.months import parse_month_range_args

BASE_URL = "https://d37ci6vzurychx.cloudfront.net"
ZONE_LOOKUP_PATH = RAW_DIR / "taxi_zone_lookup.csv"


def download_file(url: str, path: Path) -> Path:
    """Скачивает файл, если его ещё нет на диске."""
    if path.exists():
        print(f"{path} уже скачан")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    # Сначала качаем во временный файл и только потом переименовываем его.
    # Если загрузка оборвётся, на месте останется файл .part, а не обрезанный
    # файл, который при следующем запуске был бы принят за скачанный.
    tmp_path = path.with_suffix(".part")
    print(f"Скачиваю {url}")
    urlretrieve(url, tmp_path)
    tmp_path.replace(path)
    return path


def download_month(month: str) -> Path:
    return download_file(f"{BASE_URL}/trip-data/yellow_tripdata_{month}.parquet", raw_path(month))


def download_zone_lookup() -> Path:
    """Справочник зон TLC с названиями и районами города."""
    return download_file(f"{BASE_URL}/misc/taxi_zone_lookup.csv", ZONE_LOOKUP_PATH)


def main():
    months = parse_month_range_args("Загрузка данных TLC по месяцам")
    download_zone_lookup()
    for month in months:
        download_month(month)


if __name__ == "__main__":
    main()