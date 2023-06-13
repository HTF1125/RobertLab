"""ROBERT"""
from typing import Optional
import pandas as pd
from pkg.src.data import get_prices

__all__ = [
    "Global64",
    "UnitedStates64",
]


class Benchmark:

    def __init__(
        self,
        prices: pd.DataFrame,
        allocations: pd.Series,
        name: Optional[str] = None,
    ) -> None:
        self.name = name
        self.prices = prices.ffill().dropna()
        self.allocations = allocations

    def __repr__(self) -> str:
        details = "; ".join(
            [f"{asset}: {weight:.2%}" for asset, weight in self.allocations.items()]
        )
        return f"Benchmark ({details})"

    def __str__(self) -> str:
        if self.name is not None:
            return self.name
        return self.__repr__()

    def performance(
        self,
        initial_investment: int = 10_000,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        prices = self.prices.copy()
        if start is not None:
            prices = prices.loc[start:]
        if end is not None:
            prices = prices.loc[:end]
        weights = pd.Series(self.allocations)
        perf = (
            prices.pct_change()
            .fillna(0)
            .dot(weights)
            .add(1)
            .cumprod()
            .multiply(initial_investment)
        )
        perf.name = self.__str__()
        return perf


class Global64(Benchmark):
    @classmethod
    def instance(cls) -> Benchmark:
        name = "global64"
        allocations = pd.Series({"ACWI": 0.6, "BND": 0.4})
        prices = get_prices(tickers=list(allocations.keys())).ffill().dropna()
        return cls(prices=prices, allocations=allocations, name=name)


class UnitedStates64(Benchmark):
    @classmethod
    def instance(cls) -> Benchmark:
        name = "US64"
        allocations = pd.Series({"SPY": 0.6, "AGG": 0.4})
        prices = get_prices(tickers=list(allocations.keys())).ffill().dropna()
        return cls(prices=prices, allocations=allocations, name=name)


