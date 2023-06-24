"""ROBERT"""
from typing import Union, List, Set, Tuple
import pandas as pd
from src.backend.core import metrics
from . import single


class MultiFactors(dict):
    def __init__(
        self,
        tickers: Union[str, List, Set, Tuple],
        factors: Tuple[str],
    ) -> None:
        for factor in factors:
            self[factor] = getattr(single, factor)(tickers=tickers)

    @property
    def standard_percentile(self) -> pd.DataFrame:
        return (
            pd.concat(
                objs=[factor.standard_percentile.stack() for _, factor in self.items()],
                axis=1,
            )
            .mean(axis=1)
            .unstack()
        ).apply(metrics.to_standard_percentile, axis=1)
