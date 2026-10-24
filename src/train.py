import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.baseline import PairMedianBaseline
from src.features import TARGET, load_month
from src.lgbm_model import LGBMDurationModel

SEED = 42
PARAMS_PATH = Path("configs/lgbm_params.json")
MODELS_DIR = Path("models")


def mae(df: pd.DataFrame, pred) -> float:
    return float(mean_absolute_error(df[TARGET], pred))


def main():
    jan, feb, test = load_month("2024-01"), load_month("2024-02"), load_month("2024-03")
    train = pd.concat([jan, feb], ignore_index=True)
    print(f"train: {len(train):,} поездок, test: {len(test):,}")

    # бейзлайны
    mae_const = mae(test, np.full(len(test), train[TARGET].median()))
    baseline = PairMedianBaseline().fit(train)
    mae_pair = mae(test, baseline.predict(test))

    # LightGBM
    params = json.loads(PARAMS_PATH.read_text())["params"]

    # 1) число деревьев: январь -> ранняя остановка по февралю
    probe = LGBMDurationModel(n_estimators=5000, random_state=SEED, **params)
    probe.fit(jan, valid=feb)
    n_trees = probe.best_iteration_

    # 2) финальная модель: январь + февраль, фиксированное число деревьев
    model = LGBMDurationModel(n_estimators=n_trees, random_state=SEED, **params)
    model.fit(train)
    mae_lgbm = mae(test, model.predict(test))

    print(f"MAE, общая медиана:       {mae_const:.2f} мин")
    print(f"MAE, медиана по паре зон: {mae_pair:.2f} мин")
    print(f"MAE, LightGBM:            {mae_lgbm:.2f} мин  ({n_trees} деревьев)")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(baseline, MODELS_DIR / "baseline.joblib")
    joblib.dump(model, MODELS_DIR / "lgbm.joblib")
    metrics = {"mae_const": mae_const, "mae_pair": mae_pair,
               "mae_lgbm": mae_lgbm, "n_trees": n_trees}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()