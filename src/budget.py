"""Эксперимент с бюджетом сложности модели.

Обучаем модель на годовом окне с несколькими темпами обучения и смотрим, как MAE
и скорость предсказаний зависят от числа деревьев. Для сравнения считаем те же
величины для текущей модели champion.
"""
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.features import TARGET, load_sample, make_model_input
from src.monitor import MODEL_ALIAS, MODEL_NAME, log
from src.months import month_range
from src.training import train_on_window

# Обучение на 2024-01 .. 2024-11, ранняя остановка по 2024-12.
WINDOW = month_range("2024-01", "2024-12")
# Январь лежит в сезоне, на котором обучалась старая модель, май вне его.
TEST_MONTHS = ["2025-01", "2025-05"]

TRAIN_PER_MONTH = 100_000
VALID_SIZE = 200_000
TEST_SIZE = 100_000
LEARNING_RATES = [0.05, 0.1, 0.2]
MAX_TREES = 1000
TREE_COUNTS = [50, 100, 200, 300, 500, 1000]
SPEED_ROWS = 50_000  # на стольких строках измеряется скорость предсказаний
EXPERIMENT = "taxi-eta-budget"
REPORT_PATH = Path("reports/budget.csv")


def evaluate(predict, tests: dict[str, pd.DataFrame]) -> dict[str, float]:
    """MAE на каждом тестовом месяце и скорость предсказаний в строках в секунду."""
    row = {}
    for month, df in tests.items():
        pred = predict(make_model_input(df))
        row[f"mae_{month}"] = float(np.abs(df[TARGET].to_numpy() - pred).mean())

    sample = make_model_input(next(iter(tests.values())).head(SPEED_ROWS))
    started = time.perf_counter()
    predict(sample)
    row["rows_per_sec"] = SPEED_ROWS / (time.perf_counter() - started)
    return row


def main():
    log(f"Читаю тестовые месяцы {', '.join(TEST_MONTHS)}")
    tests = {month: load_sample([month], TEST_SIZE) for month in TEST_MONTHS}

    rows = []
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="complexity-budget"):
        mlflow.log_params({
            "window": f"{WINDOW[0]}..{WINDOW[-1]}",
            "test_months": ",".join(TEST_MONTHS),
            "train_per_month": TRAIN_PER_MONTH,
            "learning_rates": ",".join(map(str, LEARNING_RATES)),
            "max_trees": MAX_TREES,
        })

        # Текущая модель для сравнения. Число деревьев берём из параметров запуска, который её обучил.
        client = MlflowClient()
        version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        champion_trees = int(client.get_run(version.run_id).data.params["n_trees"])
        log(f"Оцениваю текущую модель v{version.version} ({champion_trees} деревьев)")
        champion = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version.version}")
        rows.append({"model": f"champion v{version.version}", "learning_rate": np.nan,
                     "trees": champion_trees, **evaluate(champion.predict, tests)})
        log(f"Готово, {rows[-1]['rows_per_sec']:,.0f} строк в секунду")

        for lr in LEARNING_RATES:
            log(f"Обучаю модель на окне {WINDOW[0]}..{WINDOW[-1]}, learning_rate={lr}")
            model, info = train_on_window(WINDOW, TRAIN_PER_MONTH, VALID_SIZE, lr, MAX_TREES)
            log(f"Обучена за {info['train_seconds']:.0f} с, ранняя остановка на {info['n_trees']} деревьях")

            # Одна обученная модель даёт результаты для любого числа первых деревьев.
            counts = sorted({k for k in TREE_COUNTS if k < info["n_trees"]} | {info["n_trees"]})
            for k in counts:
                row = evaluate(lambda X, k=k: model.predict(X, num_iteration=k), tests)
                rows.append({"model": "годовое окно", "learning_rate": lr, "trees": k, **row})
                maes = ", ".join(f"{m} {row[f'mae_{m}']:.3f}" for m in TEST_MONTHS)
                log(f"  {k:>4} деревьев  MAE {maes}  {row['rows_per_sec']:,.0f} строк/с")

        table = pd.DataFrame(rows)
        REPORT_PATH.parent.mkdir(exist_ok=True)
        table.to_csv(REPORT_PATH, index=False)
        mlflow.log_artifact(str(REPORT_PATH))

    print()
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    log(f"Таблица сохранена в {REPORT_PATH}")


if __name__ == "__main__":
    main()