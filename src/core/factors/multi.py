"""ROBERT"""
from typing import Union, List, Set, Tuple
import pandas as pd
from src.core import metrics
from . import single


class MultiFactors(dict):
    def __init__(
        self,
        tickers: Union[str, List, Set, Tuple],
        factors: Tuple[str],
    ) -> None:
        self.factors = factors
        self.tickers = tickers
        for factor in self.factors:
            self[factor] = getattr(single, factor)()

    def standard_scaler(
        self,
    ) -> pd.DataFrame:
        return (
            pd.concat(
                objs=[
                    factor.standard_scaler(tickers=self.tickers).stack()
                    for _, factor in self.items()
                ],
                axis=1,
            )
            .mean(axis=1)
            .unstack()
        ).apply(metrics.to_standard_scaler, axis=1)
