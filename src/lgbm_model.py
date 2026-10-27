import lightgbm as lgb
import pandas as pd

from src.features import CAT_FEATURES, FEATURES, TARGET

# Признаки моделей, обученных до появления признака месяца. Такие модели не хранят
# свой список признаков, поэтому для них он задан здесь.
LEGACY_FEATURES = ["PULocationID", "DOLocationID", "pickup_hour", "pickup_dow"]


class LGBMDurationModel:
    """LightGBM с объективом L1; тот же интерфейс fit/predict, что у бейзлайна."""

    def __init__(self, **params):
        self.params = params

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None,
            early_stopping_rounds: int = 50) -> "LGBMDurationModel":
        # Модель запоминает, на каких признаках обучалась. Список признаков в коде
        # со временем меняется, а сохранённая в реестре модель должна работать как раньше.
        self.features_ = list(FEATURES)
        cat_features = [col for col in CAT_FEATURES if col in self.features_]

        self.model_ = lgb.LGBMRegressor(objective="l1", verbose=-1, **self.params)
        fit_kwargs = {"categorical_feature": cat_features}
        if valid is not None:
            fit_kwargs["eval_X"] = (valid[self.features_],)
            fit_kwargs["eval_y"] = (valid[TARGET],)
            fit_kwargs["eval_metric"] = "l1"
            fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        self.model_.fit(train[self.features_], train[TARGET], **fit_kwargs)
        return self

    @property
    def best_iteration_(self) -> int:
        return self.model_.best_iteration_

    @property
    def n_trees_(self) -> int:
        """Сколько деревьев на самом деле в модели."""
        return self.model_.booster_.num_trees()

    def predict(self, df: pd.DataFrame, num_iteration: int | None = None):
        """Предсказание. num_iteration позволяет использовать только первые деревья."""
        features = getattr(self, "features_", LEGACY_FEATURES)
        return self.model_.predict(df[features], num_iteration=num_iteration)