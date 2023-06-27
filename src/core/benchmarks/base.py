"""ROBERT"""
from typing import Optional, Any
import pandas as pd
from src.core import universes, metrics

class Benchmark:
    UNIVERSE = universes.UsAllocation()
    PERFORMANCE = pd.Series(dtype=float)
    WEIGHTS = pd.DataFrame()
    INCEPTION = "1900-01-01"
    INITIAL_INVESTMENT = 10_000.
    MIN_WINDOW = 0

    @property
    def inception(self) -> str:
        return self.INCEPTION

    @inception.setter
    def inception(self, inception: Optional[str]) -> None:
        if inception is None:
            return
        self.INCEPTION = inception

    @property
    def initial_investment(self) -> float:
        return self.INITIAL_INVESTMENT

    @initial_investment.setter
    def initial_investment(self, initial_investment: Optional[float]) -> None:
        if initial_investment is None:
            return
        self.INITIAL_INVESTMENT = initial_investment

    @property
    def min_window(self) -> int:
        return self.MIN_WINDOW

    @min_window.setter
    def min_window(self, min_window: Optional[int]) -> None:
        if min_window is None:
            return
        self.MIN_WINDOW = min_window

    @property
    def weights(self) -> pd.DataFrame:
        return self.WEIGHTS

    @weights.setter
    def weights(self, weights: pd.DataFrame) -> None:
        self.WEIGHTS = weights

    @property
    def performance(self) -> pd.Series:
        return self.PERFORMANCE

    @performance.setter
    def performance(self, performance: pd.Series) -> None:
        performance.name = self.__class__.__name__
        self.PERFORMANCE = performance

    def new(
        self,
        inception: Optional[str] = None,
        initial_investment: Optional[float] = None,
        min_window: Optional[int] = None,
    ) -> "Benchmark":
        self.initial_investment = initial_investment
        self.inception = inception
        self.min_window = min_window
        return self

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
        if date is not None:
            return self.performance.loc[:date]
        return self.performance

    def get_weights(self, date: Optional[Any] = None) -> pd.Series:
        if date is None:
            return self.weights.iloc[-1]
        try:
            return self.weights.loc[:date].iloc[-1]
        except IndexError:
            return self.weights.iloc[0]


    def get_alpha(self, performance: pd.Series) -> pd.Series:
        bm_performance = self.performance.reindex(performance.index).ffill()
        pri_return_1 = performance.pct_change().fillna(0)
        pri_return_2 = bm_performance.pct_change().fillna(0)
        return (pri_return_1 - pri_return_2).add(1).cumprod()


    @property
    def drawdown(self) -> pd.Series:
        return metrics.to_drawdown(self.performance)