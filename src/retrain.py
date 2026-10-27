"""Переобучение модели и решение о её выкатке по схеме champion/challenger.

Месяц t только что закончился, и длительности его поездок известны. Обучаем
претендента на 12 месяцах перед t и сравниваем его с текущей моделью на поездках
месяца t, которых не видела ни одна из моделей. Претендент становится champion,
только если он лучше с уверенностью, а не случайно.
"""
import argparse
from statistics import NormalDist

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.features import TARGET, load_month, make_model_input
from src.mlflow_model import DurationPyfunc
from src.monitor import MODEL_ALIAS, MODEL_NAME, SAMPLE_SIZE, log, sample
from src.months import months_before, parse_month
from src.training import train_on_window
from src.validation import validate_month

# Бюджет сложности выбран по эксперименту src/budget.py.
LEARNING_RATE = 0.2
MAX_TREES = 100
WINDOW_MONTHS = 12
TRAIN_PER_MONTH = 100_000
VALID_SIZE = 200_000

N_BOOTSTRAP = 2000
CONFIDENCE = 0.95
# Бэктест 2024-01 .. 2025-06 показал, что пороги 1-2% заставляют модель стареть
# в ожидании накопленного выигрыша. Проверки интервалом достаточно, чтобы отсечь
# претендентов, неотличимых от текущей модели.
MIN_RELATIVE_IMPROVEMENT = 0.0

CHALLENGER_ALIAS = "challenger"
EXPERIMENT = "taxi-eta-retrain"
SEED = 42


def get_champion_version(client: MlflowClient) -> str | None:
    """Номер версии с алиасом champion или None, если такой версии ещё нет.

    Отсутствие модели проверяется явно, без перехвата исключений. Иначе ошибка связи
    с сервером выглядела бы как "модели нет", и претендент выкатился бы без сравнения.
    """
    models = client.search_registered_models(filter_string=f"name = '{MODEL_NAME}'")
    if not models:
        return None
    return models[0].aliases.get(MODEL_ALIAS)


def cluster_bootstrap(d: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Бутстрап среднего d, в котором пересэмплируются целые кластеры (дни), а не отдельные поездки.

    Возвращает N_BOOTSTRAP бутстрап-оценок среднего.
    """
    # Для каждого дня сумма d и число поездок. Дальше работаем только с этими числами.
    days = pd.DataFrame({"d": d, "day": clusters}).groupby("day")["d"].agg(["sum", "count"])
    sums, counts = days["sum"].to_numpy(), days["count"].to_numpy()
    n_days = len(days)

    # Каждая строка weights говорит, сколько раз каждый день попал в бутстрап-выборку.
    # Это то же самое, что выбрать n_days дней с возвращением.
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(n_days, np.full(n_days, 1 / n_days), size=N_BOOTSTRAP)

    # Среднее по всем поездкам выбранных дней. Дни бывают разного размера,
    # поэтому делим сумму d на сумму числа поездок, а не усредняем средние по дням.
    return (weights @ sums) / (weights @ counts)


def compare(d: np.ndarray, days: np.ndarray) -> dict[str, float]:
    """Средний выигрыш претендента и доверительные интервалы для него."""
    alpha = 1 - CONFIDENCE
    boot = cluster_bootstrap(d, days)
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])

    # Наивный интервал считает все поездки независимыми. Он нужен только для сравнения.
    z = NormalDist().inv_cdf(1 - alpha / 2)
    se = d.std(ddof=1) / np.sqrt(len(d))

    return {
        "improvement": float(d.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "naive_ci_low": float(d.mean() - z * se),
        "naive_ci_high": float(d.mean() + z * se),
        "n_days": int(len(np.unique(days))),
    }


def main():
    parser = argparse.ArgumentParser(description="Переобучение и сравнение с текущей моделью")
    parser.add_argument("month", type=parse_month, help="месяц, на котором сравниваются модели, например 2025-05")
    month = parser.parse_args().month

    window = months_before(month, WINDOW_MONTHS)
    log(f"Месяц сравнения {month}, окно обучения претендента {window[0]}..{window[-1]}")

    # Сначала проверяем все нужные данные. Обучать на битых данных хуже, чем не обучать вовсе.
    for m in window + [month]:
        report = validate_month(m)
        if not report.ok:
            log(f"Месяц {m} не прошёл проверку: {'; '.join(report.errors)}. Переобучение отменено")
            return
    log("Все месяцы прошли проверку")

    client = MlflowClient()
    champion_version = get_champion_version(client)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"retrain-{month}"):
        log(f"Обучаю претендента, learning_rate={LEARNING_RATE}, до {MAX_TREES} деревьев")
        challenger, info = train_on_window(window, TRAIN_PER_MONTH, VALID_SIZE, LEARNING_RATE, MAX_TREES)
        log(f"Обучен за {info['train_seconds']:.0f} с, деревьев {info['n_trees']}")
        mlflow.log_params({**info, "eval_month": month, "champion_version": champion_version or "нет",
                           "min_relative_improvement": MIN_RELATIVE_IMPROVEMENT})

        df = sample(load_month(month), SAMPLE_SIZE)
        X = make_model_input(df)
        y = df[TARGET].to_numpy()
        err_challenger = np.abs(y - challenger.predict(X))
        metrics = {"mae_challenger": float(err_challenger.mean()), "n_sample": len(df)}

        if champion_version is None:
            log("Текущей модели нет, претендент становится champion без сравнения")
            promote = True
        else:
            log(f"Делаю предсказания текущей модели v{champion_version} на {len(df):,} поездках")
            champion = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{champion_version}")
            err_champion = np.abs(y - champion.predict(X))

            # Положительная разность значит, что на этой поездке претендент ошибся меньше.
            d = err_champion - err_challenger
            days = df["tpep_pickup_datetime"].dt.day.to_numpy()
            stats = compare(d, days)
            metrics.update({"mae_champion": float(err_champion.mean()), **stats})

            threshold = MIN_RELATIVE_IMPROVEMENT * metrics["mae_champion"]
            metrics["threshold"] = threshold
            promote = stats["ci_low"] > threshold
            log(f"MAE champion {metrics['mae_champion']:.3f}, challenger {metrics['mae_challenger']:.3f}")
            log(f"Выигрыш претендента {stats['improvement']:+.3f} мин, "
                f"{CONFIDENCE:.0%} интервал по дням [{stats['ci_low']:+.3f}, {stats['ci_high']:+.3f}], "
                f"наивный интервал [{stats['naive_ci_low']:+.3f}, {stats['naive_ci_high']:+.3f}]")
            log(f"Порог выкатки {threshold:.3f} мин ({MIN_RELATIVE_IMPROVEMENT:.0%} от MAE champion)")

        mlflow.log_metrics(metrics)
        mlflow.log_param("decision", "promoted" if promote else "rejected")

        log("Регистрирую претендента в реестре")
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=DurationPyfunc(challenger),
            input_example=X.head(5),
            code_paths=["src"],
            pip_requirements="requirements.txt",
            registered_model_name=MODEL_NAME,
        )
        version = model_info.registered_model_version
        client.set_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS, version)
        client.set_model_version_tag(MODEL_NAME, version, "eval_month", month)
        client.set_model_version_tag(MODEL_NAME, version, "decision", "promoted" if promote else "rejected")

        if promote:
            client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
            log(f"Решение: претендент v{version} становится champion")
        else:
            log(f"Решение: остаётся champion v{champion_version}, претендент v{version} отклонён")


if __name__ == "__main__":
    main()