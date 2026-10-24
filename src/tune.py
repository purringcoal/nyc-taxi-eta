import json
from pathlib import Path

import optuna
from sklearn.metrics import mean_absolute_error

from src.features import TARGET, load_month
from src.lgbm_model import LGBMDurationModel

SEED = 42
N_TRIALS = 30
TRAIN_SAMPLE = 1_000_000
VALID_SAMPLE = 300_000
PARAMS_PATH = Path("configs/lgbm_params.json")


def main():
    train = load_month("2024-01").sample(TRAIN_SAMPLE, random_state=SEED)
    valid = load_month("2024-02").sample(VALID_SAMPLE, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 512, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 2000, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
        }
        model = LGBMDurationModel(n_estimators=2000, random_state=SEED, **params)
        model.fit(train, valid=valid)
        trial.set_user_attr("n_trees", model.best_iteration_)
        return mean_absolute_error(valid[TARGET], model.predict(valid))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"Лучшая MAE на валидации: {study.best_value:.3f} мин")
    print(f"Параметры: {study.best_params}")

    PARAMS_PATH.parent.mkdir(exist_ok=True)
    result = {
        "params": study.best_params,
        "valid_mae": study.best_value,
        "n_trees_on_sample": study.best_trial.user_attrs["n_trees"],
    }
    PARAMS_PATH.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()