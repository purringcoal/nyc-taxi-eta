import numpy as np
import pandas as pd
import pytest

from src import lgbm_model
from src.features import FEATURES, TARGET
from src.lgbm_model import LEGACY_FEATURES, LGBMDurationModel


def make_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "PULocationID": rng.integers(1, 266, n),
        "DOLocationID": rng.integers(1, 266, n),
        "pickup_hour": rng.integers(0, 24, n),
        "pickup_dow": rng.integers(0, 7, n),
        "pickup_month": rng.integers(1, 13, n),
    })
    df[TARGET] = 5 + df["pickup_hour"] / 3 + rng.exponential(2, n)
    return df


@pytest.fixture(scope="module")
def model():
    return LGBMDurationModel(n_estimators=20, learning_rate=0.2).fit(make_data())


def test_extra_and_shuffled_columns_do_not_change_predictions(model):
    df = make_data(100, seed=1)
    expected = model.predict(df[FEATURES])
    messy = df[list(reversed(df.columns))].assign(extra=123)
    np.testing.assert_allclose(model.predict(messy), expected)


def test_model_keeps_copy_of_feature_list(model):
    assert model.features_ == FEATURES
    assert model.features_ is not FEATURES


def test_number_of_trees_without_validation(model):
    assert model.n_trees_ == 20


def test_num_iteration_uses_first_trees(model):
    df = make_data(100, seed=1)
    assert not np.allclose(model.predict(df, num_iteration=5), model.predict(df))


def test_legacy_model_without_feature_list(monkeypatch):
    # Модель, обученная до появления признака месяца, не хранит список признаков.
    monkeypatch.setattr(lgbm_model, "FEATURES", LEGACY_FEATURES)
    old = LGBMDurationModel(n_estimators=5).fit(make_data())
    del old.features_
    preds = old.predict(make_data(10, seed=2))  # на входе пять признаков, модель знает четыре
    assert preds.shape == (10,)