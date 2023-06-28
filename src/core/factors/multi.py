"""ROBERT"""
from typing import Union, List, Set, Tuple, Optional
import pandas as pd
from src.core import metrics
from . import base


class MultiFactors(dict):
    factor = pd.DataFrame()

    def __init__(
        self,
        factors: Tuple[Union[str, base.Factor]] = tuple(),
    ) -> None:
        for factor in factors:
            self.add_factor(factor)

    def add_factor(self, factor: Union[str, base.Factor]) -> "MultiFactors":
        if isinstance(factor, str):
            factor = base.get_attr(factor)
        self[factor.__class__.__name__] = factor
        return self

    def __getitem__(self, name: str) -> base.Factor:
        return super().__getitem__(name)

    def items(self) -> List[Tuple[str, base.Factor]]:
        return [(name, factor) for name, factor in super().items()]

    def compute(self, tickers: Union[str, List, Set, Tuple]) -> "MultiFactors":
        factortoconcat = []
        for _, factor in self.items():
            factor.compute(tickers)
            factortoconcat.append(factor.standard_scaler().stack())
        if not factortoconcat:
            return self
        self.factor = pd.concat(objs=factortoconcat, axis=1).mean(axis=1).unstack()
        return self

    def compute_standard_scaler(
        self, tickers: Union[str, List, Set, Tuple]
    ) -> "MultiFactors":
        self.compute(tickers)
        if self.factor.empty:
            return self
        self.factor = self.factor.apply(metrics.to_standard_scaler, axis=1)
        return self

    def get_factor_by_date(
        self, date: Optional[Union[str, pd.Timestamp]] = None
    ) -> pd.Series:
        if date is None:
            return self.factor.iloc[-1]
        return self.factor.loc[:date].iloc[-1]
