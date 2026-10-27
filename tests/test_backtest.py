import numpy as np
import pandas as pd
import pytest

from src.backtest import always_fresh, always_lagged, never, rule

MONTHS = ["2025-01", "2025-02", "2025-03", "2025-04"]


def test_simple_strategies():
    assert never(MONTHS) == dict.fromkeys(MONTHS, "2025-01")
    assert always_fresh(MONTHS) == {m: m for m in MONTHS}
    assert always_lagged(MONTHS) == {"2025-01": "2025-01", "2025-02": "2025-01",
                                     "2025-03": "2025-02", "2025-04": "2025-03"}


class FakeBacktest:
    """Подмена бэктеста с заданными ошибками моделей, без обучения."""

    def __init__(self, newer_is_better: bool, n=3000):
        self.newer_is_better = newer_is_better
        rng = np.random.default_rng(0)
        self.samples = {}
        for m in MONTHS:
            t = pd.Timestamp(f"{m}-01") + pd.to_timedelta(rng.integers(0, 28 * 86400, n), unit="s")
            self.samples[m] = pd.DataFrame({"tpep_pickup_datetime": t})
        self.noise = {m: rng.normal(0, 0.1, n) for m in MONTHS}

    def abs_errors(self, model_month, month):
        # Если новые модели лучше, ошибка растёт с возрастом модели на полминуты за месяц.
        age = MONTHS.index(month) - MONTHS.index(model_month)
        base = 3.0 + (0.5 * age if self.newer_is_better else 0.0)
        return base + self.noise[month]

    def mae(self, model_month, month):
        return float(self.abs_errors(model_month, month).mean())


def test_rule_promotes_clearly_better_models():
    deployed, decisions = rule(FakeBacktest(newer_is_better=True), MONTHS, threshold=0.0)
    assert deployed == always_lagged(MONTHS)
    assert all(d["promoted"] for d in decisions)


def test_rule_keeps_champion_when_nothing_changes():
    deployed, decisions = rule(FakeBacktest(newer_is_better=False), MONTHS, threshold=0.0)
    assert deployed == never(MONTHS)
    assert not any(d["promoted"] for d in decisions)


@pytest.mark.parametrize("newer_is_better", [True, False])
def test_no_strategy_uses_model_trained_on_its_own_month(newer_is_better):
    # Главный инвариант бэктеста. Модель M[s] обучена на данных до месяца s и может
    # работать только начиная с s. Иначе бэктест подсмотрел бы ответы.
    plans = [never(MONTHS), always_fresh(MONTHS), always_lagged(MONTHS),
             rule(FakeBacktest(newer_is_better), MONTHS, threshold=0.0)[0]]
    for plan in plans:
        assert all(model_month <= month for month, model_month in plan.items())