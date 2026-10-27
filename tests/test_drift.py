import numpy as np
import pandas as pd
import pytest

from src.drift import psi_categorical, psi_continuous


def test_identical_distributions_give_zero():
    s = pd.Series(["a", "b", "b", "c"] * 100)
    assert psi_categorical(s, s) == pytest.approx(0.0)


def test_known_value():
    # Доли 0.5/0.5 против 0.9/0.1, посчитано по формуле вручную.
    reference = pd.Series(["x"] * 50 + ["y"] * 50)
    current = pd.Series(["x"] * 90 + ["y"] * 10)
    expected = 0.4 * np.log(0.9 / 0.5) + (-0.4) * np.log(0.1 / 0.5)
    assert psi_categorical(reference, current) == pytest.approx(expected)


def test_psi_is_symmetric():
    a = pd.Series(["x"] * 70 + ["y"] * 30)
    b = pd.Series(["x"] * 40 + ["y"] * 60)
    assert psi_categorical(a, b) == pytest.approx(psi_categorical(b, a))


def test_category_missing_in_reference_gives_finite_value():
    psi = psi_categorical(pd.Series(["x"] * 100), pd.Series(["x"] * 90 + ["new"] * 10))
    assert np.isfinite(psi) and psi > 0


def test_normal_shift_matches_theory():
    # Для N(0,1) и N(mu,1) симметризованная KL-дивергенция равна mu^2. Разбиение на корзины
    # теряет информацию, поэтому PSI немного меньше, но близок к теории.
    rng = np.random.default_rng(0)
    psi = psi_continuous(rng.normal(0, 1, 200_000), rng.normal(0.5, 1, 200_000))
    assert 0.2 < psi < 0.25