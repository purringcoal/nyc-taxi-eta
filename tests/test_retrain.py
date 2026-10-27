from types import SimpleNamespace

import numpy as np
import pytest

from src import retrain
from src.retrain import cluster_bootstrap, compare, get_champion_version


def days_for(n, n_days=30, seed=0):
    return np.random.default_rng(seed).integers(1, n_days + 1, n)


def test_no_difference_gives_interval_around_zero():
    rng = np.random.default_rng(1)
    d = rng.normal(0, 2, 50_000)
    stats = compare(d, days_for(len(d)))
    assert stats["ci_low"] < 0 < stats["ci_high"]


def test_clear_improvement_is_detected():
    rng = np.random.default_rng(2)
    d = rng.normal(0.3, 2, 50_000)
    assert compare(d, days_for(len(d)))["ci_low"] > 0


def test_day_effects_widen_the_interval():
    # Если у каждого дня свой общий сдвиг, интервал по дням должен быть заметно шире наивного.
    rng = np.random.default_rng(3)
    days = days_for(300_000)
    day_shift = rng.normal(0, 0.1, 31)
    d = 0.05 + day_shift[days] + rng.normal(0, 2, len(days))
    stats = compare(d, days)
    cluster_width = stats["ci_high"] - stats["ci_low"]
    naive_width = stats["naive_ci_high"] - stats["naive_ci_low"]
    assert cluster_width > 3 * naive_width


def test_bootstrap_is_centered_on_mean():
    rng = np.random.default_rng(4)
    d = rng.normal(1, 1, 10_000)
    boot = cluster_bootstrap(d, days_for(len(d)))
    assert len(boot) == retrain.N_BOOTSTRAP
    assert boot.mean() == pytest.approx(d.mean(), abs=0.02)


class FakeClient:
    """Подмена клиента MLflow. Возвращает заданный список моделей или бросает заданное исключение."""

    def __init__(self, models=None, error=None):
        self.models, self.error = models or [], error

    def search_registered_models(self, filter_string):
        if self.error:
            raise self.error
        return self.models


def test_no_registered_model():
    assert get_champion_version(FakeClient()) is None


def test_model_without_champion_alias():
    assert get_champion_version(FakeClient([SimpleNamespace(aliases={"challenger": "4"})])) is None


def test_champion_alias():
    model = SimpleNamespace(aliases={"champion": "3", "challenger": "4"})
    assert get_champion_version(FakeClient([model])) == "3"


def test_connection_error_is_not_mistaken_for_missing_model():
    # Если ошибку связи принять за "модели нет", претендент выкатится без сравнения.
    with pytest.raises(ConnectionError):
        get_champion_version(FakeClient(error=ConnectionError("сервер недоступен")))