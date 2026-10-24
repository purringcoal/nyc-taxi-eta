# src/lgbm_model.py
import lightgbm as lgb
import pandas as pd

from src.features import CAT_FEATURES, FEATURES, TARGET


class LGBMDurationModel:
    """LightGBM с объективом L1; тот же интерфейс fit/predict, что у бейзлайна."""

    def __init__(self, **params):
        self.params = params

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None,
            early_stopping_rounds: int = 50) -> "LGBMDurationModel":
        self.model_ = lgb.LGBMRegressor(objective="l1", verbose=-1, **self.params)
        fit_kwargs = {"categorical_feature": CAT_FEATURES}
        if valid is not None:
            fit_kwargs["eval_X"] = (valid[FEATURES],)
            fit_kwargs["eval_y"] = (valid[TARGET],)
            fit_kwargs["eval_metric"] = "l1"
            fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        self.model_.fit(train[FEATURES], train[TARGET], **fit_kwargs)
        return self

    @property
    def best_iteration_(self) -> int:
        return self.model_.best_iteration_

    def predict(self, df: pd.DataFrame):
        return self.model_.predict(df[FEATURES])