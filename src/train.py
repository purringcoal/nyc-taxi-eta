# src/train.py
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from src.baseline import PairMedianBaseline
from src.features import TARGET, load_month, make_model_input
from src.lgbm_model import LGBMDurationModel
from src.mlflow_model import DurationPyfunc

SEED = 42
TRAIN_MONTHS = ["2024-01", "2024-02"]
TEST_MONTH = "2024-03"
PARAMS_PATH = Path("configs/lgbm_params.json")
EXPERIMENT = "taxi-eta"
MODEL_NAME = "taxi-eta"


def mae(df: pd.DataFrame, pred) -> float:
    return float(mean_absolute_error(df[TARGET], pred))


def main():
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="train-lgbm"):
        jan = load_month(TRAIN_MONTHS[0])
        feb = load_month(TRAIN_MONTHS[1])
        test = load_month(TEST_MONTH)
        train = pd.concat([jan, feb], ignore_index=True)
        print(f"train: {len(train):,} поездок, test: {len(test):,}")

        # бейзлайны
        mae_const = mae(test, np.full(len(test), train[TARGET].median()))
        baseline = PairMedianBaseline().fit(train)
        mae_pair = mae(test, baseline.predict(test))

        # LightGBM
        params = json.loads(PARAMS_PATH.read_text())["params"]

        probe = LGBMDurationModel(n_estimators=5000, random_state=SEED, **params)
        probe.fit(jan, valid=feb)
        n_trees = probe.best_iteration_

        model = LGBMDurationModel(n_estimators=n_trees, random_state=SEED, **params)
        model.fit(train)
        mae_lgbm = mae(test, model.predict(test))

        # журнал
        mlflow.log_params({
            **params,
            "n_trees": n_trees,
            "seed": SEED,
            "train_months": ",".join(TRAIN_MONTHS),
            "test_month": TEST_MONTH,
            "n_train": len(train),
            "n_test": len(test),
        })
        mlflow.log_metrics({"mae_const": mae_const, "mae_pair": mae_pair, "mae_lgbm": mae_lgbm})
        mlflow.log_artifact(str(PARAMS_PATH))

        # модель -> реестр
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=DurationPyfunc(model),
            input_example=make_model_input(test.head(5)),
            code_paths=["src"],
            registered_model_name=MODEL_NAME,
        )
        version = info.registered_model_version
        MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)

    print(f"MAE, общая медиана:       {mae_const:.2f} мин")
    print(f"MAE, медиана по паре зон: {mae_pair:.2f} мин")
    print(f"MAE, LightGBM:            {mae_lgbm:.2f} мин  ({n_trees} деревьев)")
    print(f"Зарегистрирована {MODEL_NAME} v{version}, алиас champion")


if __name__ == "__main__":
    main()