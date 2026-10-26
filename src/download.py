"""Загрузка данных жёлтого такси за диапазон месяцев."""
from pathlib import Path
from urllib.request import urlretrieve

from src.features import raw_path
from src.months import parse_month_range_args

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def download_month(month: str) -> Path:
    path = raw_path(month)
    if path.exists():
        print(f"{path} уже скачан")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/yellow_tripdata_{month}.parquet"

    # Сначала качаем во временный файл и только потом переименовываем его.
    # Если загрузка оборвётся, на месте останется файл .part, а не обрезанный
    # .parquet, который при следующем запуске был бы принят за скачанный.
    tmp_path = path.with_suffix(".part")
    print(f"Скачиваю {url}")
    urlretrieve(url, tmp_path)
    tmp_path.replace(path)
    return path


def main():
    for month in parse_month_range_args("Загрузка данных TLC по месяцам"):
        download_month(month)


if __name__ == "__main__":
    main()