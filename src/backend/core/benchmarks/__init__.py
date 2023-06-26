"""ROBERT"""
from typing import Optional, Any
import pandas as pd
from src.backend.core import universes
class Benchmark:
    UNIVERSE = universes.UsAllocation()
    INCEPTION_DATE = "1900-01-01"

    def __init__(
        self,
        prices: Optional[pd.DataFrame] = None,
        min_window: int = 0,
        initial_investment: int = 10_000,
    ) -> None:
        prices = prices or self.UNIVERSE.get_prices()
        for ticker in self.UNIVERSE.get_tickers():
            if ticker not in prices.columns:
                raise ValueError(
                    f"""
                    {ticker} not in prices provided.\n
                    Required assets: {self.UNIVERSE.get_tickers()}
                    """
                )
        prices = prices.dropna(how="all").ffill()
        for asset in prices:
            prices[asset] = prices[asset].iloc[min_window:]
        self.weights = self.calculate_weights(prices=prices)
        self.initial_investment = initial_investment
        pri_return = prices.loc[self.weights.index].pct_change().fillna(0)
        bm_pri_return = pri_return.mul(self.weights).sum(axis=1)
        self.performance = (bm_pri_return.add(1).cumprod()) * self.initial_investment
        self.performance.name = "Performance"

    def calculate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = prices.copy().dropna()
        w["SPY"] = 0.6
        w["AGG"] = 0.4
        return w

    def get_performance(self, date: Optional[Any] = None) -> pd.Series:
        if date is not None:
            return self.performance.loc[:date]
        return self.performance

    def get_weights(self, date: Optional[Any] = None) -> pd.Series:
        if date is None:
            return self.weights.iloc[-1]
        return self.weights.loc[:date].iloc[-1]



class UnitedStates64(Benchmark):
    UNIVERSE = universes.UsAllocation()

    def calculate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = prices.copy().dropna()
        w["SPY"] = 0.6
        w["AGG"] = 0.4
        return w

class Global64(Benchmark):
    UNIVERSE = universes.GlobalAllocation()
    def calculate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = prices.copy().dropna()
        w["ACWI"] = 0.6
        w["BND"] = 0.4
        return w


class UnitedStates55(Benchmark):
    UNIVERSE = universes.UsAllocation()

    def calculate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = prices.copy().dropna()
        w["SPY"] = 0.5
        w["AGG"] = 0.5
        return w

class EqualWeightBenchmark(Benchmark):

    def calculate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = prices.dropna(how="all").copy()
        w = 1 / p.count(axis=1)
        w = p.mask(p.notnull(), w, axis=0)
        return w


class UnitedStatesSectorsEW(EqualWeightBenchmark):
    UNIVERSE = universes.UnitedStatesSectors()
class GlobalAssetAllocationEW(EqualWeightBenchmark):
    UNIVERSE = universes.GlobalAssetAllocation()


__all__ = [
    "UnitedStates55",
    "UnitedStates64",
    "Global64",
    "UnitedStatesSectorsEW",
    "GlobalAssetAllocationEW",
]
