"""ROBERT"""
from typing import Optional, Any
import pandas as pd
from src.core import universes, metrics


class Benchmark:
    UNIVERSE = universes.UsAllocation()
    PERFORMANCE = pd.Series(dtype=float)
    WEIGHTS = pd.DataFrame()
    INCEPTION = "1900-01-01"
    INITIAL_INVESTMENT = 10_000.0
    MIN_WINDOW = 0

    def __init__(
        self,
        inception: str = "1900-01-01",
        initial_investment: int = 10_000,
        min_window: int = 0,
    ) -> None:
        self.inception = inception
        self.initial_investment = initial_investment
        self.min_window = min_window
        self.weights = pd.DataFrame()
        self.performance = pd.Series(dtype=float)

    def compute(self, prices: Optional[pd.DataFrame] = None) -> "Benchmark":
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
            prices[asset] = prices[asset].iloc[self.min_window :]
        prices = prices.loc[self.inception :]
        self.weights = self.calculate_weights(prices=prices)
        pri_return = prices.loc[self.weights.index].pct_change().fillna(0)
        bm_pri_return = pri_return.mul(self.weights).sum(axis=1)
        self.performance = bm_pri_return.add(1).cumprod() * self.initial_investment
        return self

    def calculate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = prices.copy().dropna()
        w["SPY"] = 0.6
        w["AGG"] = 0.4
        return w

    def get_performance(self, date: Optional[Any] = None) -> pd.Series:

        if self.performance.empty:
            self.compute()
        if date is not None:
            return self.performance.loc[:date]
        return self.performance


    def get_weights(self, date: Optional[Any] = None) -> pd.Series:
        if self.weights.empty:
            self.compute()
        if date is None:
            return self.weights.iloc[-1]
        try:
            return self.weights.loc[:date].iloc[-1]
        except IndexError:
            return self.weights.iloc[0]

    def get_alpha(self, performance: pd.Series) -> pd.Series:
        if self.performance.empty:
            self.compute()
        bm_performance = self.performance.reindex(performance.index).ffill()
        return performance * self.INITIAL_INVESTMENT - bm_performance

    @property
    def drawdown(self) -> pd.Series:
        return metrics.to_drawdown(self.performance)
