"""Population Stability Index (PSI) для категориальных и непрерывных величин."""
import numpy as np
import pandas as pd

# Пустая категория дала бы логарифм нуля. Поэтому доли ограничиваются снизу малым числом.
EPS = 1e-6


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI для дискретной величины. Каждое значение считается отдельной категорией."""
    q = reference.value_counts(normalize=True)  # доли в эталоне
    p = current.value_counts(normalize=True)  # доли в новых данных

    # Выравниваем по общему набору категорий. Категория, которой нет
    # в одной из выборок, получает в ней долю 0.
    q, p = q.align(p, fill_value=0.0)
    q, p = q.clip(lower=EPS), p.clip(lower=EPS)
    return float(((p - q) * np.log(p / q)).sum())


def psi_continuous(reference, current, n_bins: int = 10) -> float:
    """PSI для непрерывной величины. Корзины строятся по квантилям эталона."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # Внутренние границы корзин делят эталон на n_bins примерно равных частей.
    # Если у величины много одинаковых значений, часть границ совпадает, и дубли убираем.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1]))

    # Номер корзины для каждого значения. Значения левее первой границы попадают
    # в корзину 0, правее последней в корзину len(edges).
    ref_bins = np.searchsorted(edges, reference, side="right")
    cur_bins = np.searchsorted(edges, current, side="right")
    return psi_categorical(pd.Series(ref_bins), pd.Series(cur_bins))