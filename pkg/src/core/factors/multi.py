"""ROBERT"""
from typing import Union, List, Set, Tuple
import pandas as pd
from pkg.src.core import metrics
from .single import Factors


class MultiFactors:
    def __init__(
        self, tickers: Union[str, List, Set, Tuple],
        factors: List[Factors],
    ) -> None:
        data = [
            factor(tickers=tickers).standard_percentile.stack() for factor in factors
        ]
        self.factors = pd.concat(data, axis=1).mean(axis=1).unstack()

    @property
    def standard_percentile(self) -> pd.DataFrame:
        return self.factors.apply(metrics.to_standard_percentile, axis=1)
