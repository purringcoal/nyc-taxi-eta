import mlflow
import pandas as pd


class DurationPyfunc(mlflow.pyfunc.PythonModel):
    """Упаковывает любую нашу модель с методом predict(df) в формат MLflow pyfunc."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame, params=None):
        return self.model.predict(model_input)