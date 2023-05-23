"""ROBERT"""
from typing import Optional, Callable, Dict
import pandas as pd


class Strategy:
    """base strategy"""

    def __init__(
        self,
        prices: pd.DataFrame,
        rebalance: Callable,
        frequency: str = "M",
        initial_investment: float = 10000.0,
        start: Optional[str] = None,
        end: Optional[str] = None,
        commission: int = 10,
        shares_frac: Optional[int] = None,
    ) -> None:
        self.total_prices: pd.DataFrame = prices.ffill()
        self.date: pd.Timestamp = pd.Timestamp(str(self.total_prices.index[0]))
        self.commission = commission
        self.rebalance: Callable = rebalance
        self.shares_frac = shares_frac
        self.cash = initial_investment
        self.shares = pd.Series(dtype=float)

        self.records: Dict = {
            "value": {},
            "cash": {},
            "shares": {},
            "capitals": {},
            "weights": {},
            "trades": {},
        }
        self.simulate(
            start=start or str(self.total_prices.index[0]),
            end=end or str(self.total_prices.index[-1]),
            freq=frequency,
        )

    ################################################################################

    @property
    def prices(self) -> pd.DataFrame:
        """prices"""
        if self.date is None:
            return pd.DataFrame()
        return self.total_prices[self.total_prices.index < self.date].dropna(
            how="all", axis=1
        )

    @property
    def value(self) -> float:
        return self.capitals.sum() + self.cash

    @property
    def capitals(self) -> pd.Series:
        return self.shares.multiply(self.total_prices.loc[self.date])

    @property
    def weights(self) -> pd.Series:
        return self.capitals.divide(self.value)

    ################################################################################

    def simulate(self, start: str, end: str, freq: str = "M") -> None:
        allocations = pd.Series(dtype=float)
        self.rebalance_dates = pd.DatetimeIndex([start]).append(
            pd.date_range(start=start, end=end, freq=freq, inclusive="neither")
        )
        self.rebalance_dates = self.rebalance_dates.append(pd.DatetimeIndex([end]))

        for self.date in self.total_prices.loc[start:end].index:
            if self.date >= self.rebalance_dates[0] or self.shares.empty:
                allocations = self.rebalance(strategy=self)
                if not isinstance(allocations, pd.Series):
                    allocations = pd.Series(allocations, dtype=float)
                if not allocations.empty:
                    if self.date >= self.rebalance_dates[0]:
                        self.rebalance_dates = self.rebalance_dates[1:]

                    # Make trades here
                    target_capials = self.value * allocations
                    target_shares = target_capials.divide(
                        self.total_prices.loc[self.date]
                    )
                    if self.shares_frac is not None:
                        target_shares = target_shares.round(self.shares_frac)

                    trade_shares = target_shares.subtract(self.shares, fill_value=0)
                    trade_shares = trade_shares[trade_shares != 0]
                    trade_capitals = trade_shares.multiply(
                        self.total_prices.loc[self.date]
                    )
                    trade_capitals += trade_capitals.multiply(
                        self.commission / 1_000
                    )
                    self.cash -= trade_capitals.sum()
                    self.shares = target_shares
            self.records["value"][self.date] = self.value
            self.records["shares"][self.date] = self.shares
            self.records["cash"][self.date] = self.cash


import yfinance as yf
from core.portfolios import Optimizer


def func(strategy: Strategy) -> pd.Series:
    return Optimizer.from_prices(prices=strategy.prices).uniform_allocation()


p = yf.download("SPY, AGG")["Adj Close"].dropna().round(4)
strategy = Strategy(prices=p, rebalance=func, start="2005-1-1", shares_frac=0)
# pd.Series(strategy.records["value"]).plot()
print(pd.DataFrame(strategy.records["shares"]).T)