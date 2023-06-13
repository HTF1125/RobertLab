"""ROBERT"""
from typing import Union, List, Set, Tuple
import pandas as pd
from pkg.src.core import metrics, factors


class MultiFactors:
    def __init__(
        self,
        tickers: Union[str, List, Set, Tuple],
        factor_list: List[str],
    ) -> None:
        data = [
            getattr(factors.single, factor)(tickers=tickers).standard_percentile.stack()
            if isinstance(factor, str)
            else factor(tickers=tickers).standard_percentile.stack()
            for factor in factor_list
        ]
        self.factors = pd.concat(data, axis=1).mean(axis=1).unstack()

    @property
    def standard_percentile(self) -> pd.DataFrame:
        return self.factors.apply(metrics.to_standard_percentile, axis=1)
