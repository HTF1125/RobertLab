"""ROBERT"""
from typing import Optional, Callable
import pandas as pd
from ..analytics import metrics


class DataStore(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

    @property
    def value(self) -> pd.Series:
        return pd.Series(self["value"])

    @property
    def cash(self) -> pd.Series:
        return pd.Series(self["cash"])

    @property
    def weights(self) -> pd.DataFrame:
        return pd.DataFrame(self["weights"]).T

    @property
    def allocations(self) -> pd.DataFrame:
        return pd.DataFrame(self["allocations"]).T

    @property
    def trades(self) -> pd.DataFrame:
        return pd.DataFrame(self["trades"]).T


class Strategy:
    """base strategy"""

    def __init__(
        self,
        prices: pd.DataFrame,
        rebalance: Callable,
        frequency: str = "M",
        start: Optional[str] = None,
        end: Optional[str] = None,
        initial_investment: float = 10_000.0,
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

        self.data = DataStore()

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
        rebalance_dates = pd.DatetimeIndex([start]).append(
            pd.date_range(start=start, end=end, freq=freq, inclusive="neither")
        )
        rebalance_dates = rebalance_dates.append(pd.DatetimeIndex([end]))

        for self.date in self.total_prices.loc[start:end].index:
            if self.date >= rebalance_dates[0]:
                allocations = self.rebalance(strategy=self)
                if not isinstance(allocations, pd.Series):
                    allocations = pd.Series(allocations, dtype=float)
                if not allocations.empty:
                    self.data["allocations"][self.date] = allocations
                    rebalance_dates = rebalance_dates[1:]

                    # Make trades here
                    target_capials = self.value * allocations
                    target_shares = target_capials.divide(
                        self.total_prices.loc[self.date]
                    )
                    if self.shares_frac is not None:
                        target_shares = target_shares.round(self.shares_frac)

                    trade_shares = target_shares.subtract(self.shares, fill_value=0)
                    trade_shares = trade_shares[trade_shares != 0]
                    self.data["trades"][self.date] = trade_shares
                    trade_capitals = trade_shares.multiply(
                        self.total_prices.loc[self.date]
                    )
                    trade_capitals += trade_capitals.multiply(self.commission / 1_000)
                    self.cash -= trade_capitals.sum()
                    self.shares = target_shares
            self.data["value"][self.date] = self.value
            self.data["shares"][self.date] = self.shares
            self.data["cash"][self.date] = self.cash
            self.data["weights"][self.date] = self.weights

    @property
    def analytics(self) -> pd.Series:
        """analytics"""
        return pd.Series(
            data={
                "Start": metrics.to_start(self.data.value).strftime("%Y-%m-%d"),
                "End": metrics.to_end(self.data.value).strftime("%Y-%m-%d"),
                "AnnReturn": metrics.to_ann_return(self.data.value),
                "AnnVolatility": metrics.to_ann_volatility(self.data.value),
                "SharpeRatio": metrics.to_sharpe_ratio(self.data.value),
                "SortinoRatio": metrics.to_sortino_ratio(self.data.value),
                "MaxDrawdown": metrics.to_max_drawdown(self.data.value),
                "Skewness": metrics.to_skewness(self.data.value),
                "Kurtosis": metrics.to_kurtosis(self.data.value),
                "VaR": metrics.to_value_at_risk(self.data.value),
                "CVaR": metrics.to_conditional_value_at_risk(self.data.value),
                "TailRatio": metrics.to_tail_ratio(self.data.value),
                # "Turnover(M)": self.data.trades.resample("M").sum().sum(axis=1).mean(),
            }
        )
