"""Обучение облегчённой модели.

train_model обучает модель на готовых таблицах, train_on_window сначала собирает
таблицы из окна месяцев. Последний месяц окна используется для ранней остановки,
остальные для обучения.
"""
import json
import time
from pathlib import Path

import pandas as pd

from src.features import load_sample
from src.lgbm_model import LGBMDurationModel

PARAMS_PATH = Path("configs/lgbm_params.json")
SEED = 42


def load_params() -> dict:
    """Гиперпараметры, подобранные Optuna. Темп обучения и число деревьев задаются отдельно."""
    params = json.loads(PARAMS_PATH.read_text())["params"]
    params.pop("learning_rate", None)
    return params


def train_model(train: pd.DataFrame, valid: pd.DataFrame, learning_rate: float,
                max_trees: int) -> tuple[LGBMDurationModel, dict]:
    """Обучает модель на train с ранней остановкой по valid."""
    started = time.perf_counter()
    model = LGBMDurationModel(n_estimators=max_trees, learning_rate=learning_rate,
                              random_state=SEED, **load_params())
    model.fit(train, valid=valid)

    info = {
        "n_train": len(train),
        "learning_rate": learning_rate,
        "max_trees": max_trees,
        "n_trees": model.n_trees_,
        "train_seconds": round(time.perf_counter() - started, 1),
    }
    return model, info


def train_on_window(window: list[str], train_per_month: int, valid_size: int,
                    learning_rate: float, max_trees: int) -> tuple[LGBMDurationModel, dict]:
    """Обучает модель на всех месяцах окна, кроме последнего, и останавливается по последнему.

    Возвращает модель и словарь с информацией об обучении для журнала.
    """
    train_months, valid_month = window[:-1], window[-1]
    train = load_sample(train_months, train_per_month, seed=SEED)
    valid = load_sample([valid_month], valid_size, seed=SEED)

    model, info = train_model(train, valid, learning_rate, max_trees)
    return model, {"train_months": ",".join(train_months), "valid_month": valid_month, **info}