import pandas as pd

from src.features import PAIR, TARGET


class PairMedianBaseline:
    """Медиана длительности по паре зон, для новых пар - общая медиана."""

    def fit(self, df: pd.DataFrame) -> "PairMedianBaseline":
        self.global_median_ = float(df[TARGET].median())
        self.table_ = df.groupby(PAIR)[TARGET].median() 
        return self

    def predict(self, df: pd.DataFrame):
        idx = pd.MultiIndex.from_frame(df[PAIR])
        return self.table_.reindex(idx).fillna(self.global_median_).to_numpy()
    