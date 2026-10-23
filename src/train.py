# src/train.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.baseline import PairMedianBaseline
from src.features import TARGET, load_month

TRAIN_MONTHS = ["2024-01", "2024-02"]
VAL_MONTH = "2024-03"
MODELS_DIR = Path("models")


def main():
    train = pd.concat([load_month(m) for m in TRAIN_MONTHS], ignore_index=True)
    val = load_month(VAL_MONTH)
    print(f"train: {len(train):,} поездок, val: {len(val):,}")

    # бейзлайн 0: константа
    const_pred = np.full(len(val), train[TARGET].median())
    mae_const = mean_absolute_error(val[TARGET], const_pred)

    # бейзлайн 1: медиана по паре зон
    model = PairMedianBaseline().fit(train)
    mae_pair = mean_absolute_error(val[TARGET], model.predict(val))

    print(f"MAE, общая медиана:      {mae_const:.2f} мин")
    print(f"MAE, медиана по паре зон: {mae_pair:.2f} мин")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "baseline.joblib")
    metrics = {"mae_const": float(mae_const), "mae_pair": float(mae_pair)}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()