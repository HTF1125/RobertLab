"""ROBERT"""
from typing import Optional
import pandas as pd
from pkg.src.data import get_prices

__all__ = [
    "Global64",
    "UnitedStates64",
]

class Benchmark:
    @property
    def weights(self) -> pd.Series:
        return pd.Series(
            {
                asset: weight
                for asset, weight in self.__class__.__dict__.items()
                if not asset.startswith("__")
            }
        )

    @property
    def prices(self) -> pd.DataFrame:
        return get_prices(tickers=list(self.weights.keys())).ffill().dropna()

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
        perf = (
            prices.pct_change()
            .fillna(0)
            .dot(self.weights)
            .add(1)
            .cumprod()
            .multiply(initial_investment)
        )
        perf.name = self.__str__()
        return perf

    def __str__(self) -> str:
        return f"<Benchmark {self.weights.to_dict()}>"


class Global64(Benchmark):
    ACWI = 0.6
    BND = 0.4


class UnitedStates64(Benchmark):
    SPY = 0.6
    AGG = 0.4
