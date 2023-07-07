"""ROBERT"""
import sys
from typing import List, Union, Set, Tuple, Iterable, Type
import numpy as np
import pandas as pd
from src.core import metrics
from src.backend import data


class Factor(object):
    def __init__(self) -> None:
        self.factor = pd.DataFrame()

    # def __repr__(self) -> str:
    #     return "Base Factors"

    # def __str__(self) -> str:
    #     return "Base Factors"

    def get_factor(
        self, tickers: Union[str, List, Set, Tuple], method: str = "standard_scaler"
    ) -> pd.DataFrame:
        assert method == "standard_scaler"
        return self.standard_scaler(tickers)

    def standard_scaler(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        if self.factor.empty:
            self.fit(tickers)
        return self.factor.apply(metrics.to_standard_scaler, axis=1)

    def fit(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        raise NotImplementedError("Yout must implement `compute` method.")

    def get_factor_by_date(
        self,
        tickers: Union[str, List, Set, Tuple],
        date: str,
        method: str = "standard_scaler",
    ) -> pd.Series:
        tickers = (
            tickers
            if isinstance(tickers, (list, set, tuple))
            else tickers.replace(",", " ").split()
        )
        new_tickers = [
            ticker for ticker in tickers if ticker not in self.factor.columns
        ]
        if new_tickers:
            self.factor = pd.concat(
                objs=[self.factor, self.fit(new_tickers)], axis=1
            )
        out_factor = self.factor.ffill().loc[:date].iloc[-1]
        if method == "standard_scaler":
            return metrics.to_standard_scaler(out_factor)
        return out_factor


