from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
MONTHS = ["2024-01", "2024-02", "2024-03"]
RAW_DIR = Path("data/raw")

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for month in MONTHS:
        path = RAW_DIR / f"yellow_tripdata_{month}.parquet"
        if path.exists():
            print(f"{path} уже скачан.")
            continue
        url = f"{BASE_URL}/yellow_tripdata_{month}.parquet"
        print(f"Скачиваю {url} в {path}...")
        urlretrieve(url, path)
        print(path.stat().st_size, "байт скачано.")


if __name__ == "__main__":
    main()