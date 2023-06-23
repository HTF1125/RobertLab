"""ROBERT"""
from typing import Union, List, Set, Tuple
import pandas as pd
from pkg.src.core import metrics
from . import single


class MultiFactors:
    def __init__(
        self,
        tickers: Union[str, List, Set, Tuple],
        factors: Tuple[str],
    ) -> None:
        factor_data = []

        for factor in factors:
            if isinstance(factor, str):
                factor = getattr(single, factor)
            factor_data.append(factor(tickers=tickers).standard_percentile.stack())

        self.factors = pd.concat(factor_data, axis=1).mean(axis=1).unstack()

    @property
    def standard_percentile(self) -> pd.DataFrame:
        return self.factors.apply(metrics.to_standard_percentile, axis=1)
