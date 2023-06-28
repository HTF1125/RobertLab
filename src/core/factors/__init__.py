"""ROBERT"""
from typing import Union, List, Set, Tuple, Optional
import pandas as pd
from src.core import metrics
from .base import *

__all__ = [
    "PriceMomentum1M",
    "PriceMomentum2M",
    "PriceMomentum3M",
    "PriceMomentum6M",
    "PriceMomentum9M",
    "PriceMomentum12M",
    "PriceMomentum18M",
    "PriceMomentum24M",
    "PriceMomentum36M",
    "PriceMomentum6M1M",
    "PriceMomentum9M1M",
    "PriceMomentum12M1M",
    "PriceMomentum18M1M",
    "PriceMomentum24M1M",
    "PriceMomentum36M1M",
    "PriceMomentum6M2M",
    "PriceMomentum9M2M",
    "PriceMomentum12M2M",
    "PriceMomentum18M2M",
    "PriceMomentum24M2M",
    "PriceMomentum36M2M",
    "PriceVolatility1M",
    "PriceVolatility3M",
    "VolumeCoefficientOfVariation1M",
    "VolumeCoefficientOfVariation3M",
    "PriceRelVol1M3M",
]


def get(factor: str) -> Factor:
    # Use getattr() to get the attribute value
    try:
        return getattr(sys.modules[__name__], factor)()
    except AttributeError as exc:
        raise ValueError(f"Invalid factor: {factor}") from exc


class MultiFactors(dict):
    factor = pd.DataFrame()

    def __init__(
        self,
        factors: Tuple[Union[str, Factor]] = tuple(),
    ) -> None:
        for factor in factors:
            self.add_factor(factor)

    def add_factor(self, factor: Union[str, Factor]) -> "MultiFactors":
        if isinstance(factor, str):
            factor = get(factor)
        self[factor.__class__.__name__] = factor
        return self

    def __getitem__(self, name: str) -> Factor:
        return super().__getitem__(name)

    def items(self) -> List[Tuple[str, Factor]]:
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
