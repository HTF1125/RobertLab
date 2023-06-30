""""""
import sys
import logging
from typing import Union, List, Set, Tuple, Optional, Type
import pandas as pd
from src.core import metrics
from .base import Factor, get


logger = logging.getLogger(__name__)


class MultiFactor(dict):
    factor = pd.DataFrame()

    def __init__(
        self,
        *factors: Union[str, Factor, Type[Factor]],
    ) -> None:
        self.add_factor(*factors)

    def add_factor(self, *factors: Union[str, Factor, Type[Factor]]) -> "MultiFactor":
        for factor in factors:
            if isinstance(factor, str):
                parsed_factor = get(factor)
                self[parsed_factor.__class__.__name__] = parsed_factor
            elif isinstance(factor, type):
                if issubclass(factor, Factor):
                    self[factor.__name__] = factor()
                    continue
                logger.warning("add factor: %s failed.", factor)
            elif issubclass(factor.__class__, Factor):
                self[factor.__class__.__name__] = factor
            else:
                logger.warning("add factor: %s failed.", factor)
        return self

    def __getitem__(self, name: str) -> Factor:
        return super().__getitem__(name)

    def items(self) -> List[Tuple[str, Factor]]:
        return [(name, factor) for name, factor in super().items()]

    # def compute(self, tickers: Union[str, List, Set, Tuple]) -> "MultiFactors":
    #     factortoconcat = []
    #     for _, factor in self.items():
    #         factortoconcat.append(factor.get_factor(tickers).stack())
    #     if not factortoconcat:
    #         return self
    #     self.factor = pd.concat(objs=factortoconcat, axis=1).mean(axis=1).unstack()
    #     return self

    # def get_factor(
    #     self,
    #     tickers: Union[str, List, Set, Tuple],
    #     method: str = "standard_scaler",
    # ) -> pd.DataFrame:
    #     assert method == "standard_scaler"
    #     if self.factor.empty:
    #         self.compute(tickers)
    #     return self.factor.apply(metrics.to_standard_scaler, axis=1)

    def get_factor_by_date(
        self,
        tickers: Union[str, List, Set, Tuple],
        date: Optional[Union[str, pd.Timestamp]],
    ) -> pd.Series:
        factortoconcat = []
        for _, factor in self.items():
            base_factor = factor.get_factor_by_date(tickers, date=date)
            factortoconcat.append(base_factor)
        if not factortoconcat:
            return pd.Series(dtype=float)
        return metrics.to_standard_scaler(
            pd.concat(objs=factortoconcat, axis=1).mean(axis=1)
        )
