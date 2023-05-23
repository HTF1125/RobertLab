"""ROBERT"""
from typing import Optional, Callable
import pandas as pd
from ..analytics import metrics


class DataStore(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

    def flatten(self):
        """
        Flatten the nested data structure into a single-level dictionary.
        Returns:
            dict: A flattened dictionary.
        """
        flattened = {}

        def _flatten(dictionary, prefix=""):
            for key, value in dictionary.items():
                if isinstance(value, DataStore):
                    _flatten(value, prefix + key + ".")
                else:
                    flattened[prefix + key] = value

        _flatten(self)
        return flattened

    def load(self, data):
        """
        Load data into the DataStore.
        Args:
            data (dict): The data to be loaded.
        """
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = DataStore()
                self[key].load(value)
            else:
                self[key] = value

    def to_dict(self):
        """
        Convert the DataStore into a regular nested dictionary.
        Returns:
            dict: The nested dictionary representing the DataStore.
        """
        result = {}
        for key, value in self.items():
            if isinstance(value, DataStore):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


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
        self.initial_investment = initial_investment
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
    def value(self) -> pd.Series:
        """strategy value"""
        return pd.Series(self.data.get("value"))

    @property
    def cash(self) -> pd.Series:
        """strategy cash"""
        return pd.Series(self.data.get("cash"))

    @property
    def allocations(self) -> pd.DataFrame:
        """strategy cash"""
        return pd.DataFrame(self.data.get("allocations")).T

    ################################################################################

    def simulate(self, start: str, end: str, freq: str = "M") -> None:
        cash = self.initial_investment
        shares = pd.Series(dtype=float)
        allocations = pd.Series(dtype=float)
        rebalance_dates = pd.DatetimeIndex([start]).append(
            pd.date_range(start=start, end=end, freq=freq, inclusive="neither")
        )
        rebalance_dates = rebalance_dates.append(pd.DatetimeIndex([end]))

        for self.date in self.total_prices.loc[start:end].index:

            capitals = shares.multiply(self.total_prices.loc[self.date])
            value = capitals.sum() + cash
            weights = capitals.divide(value)

            if self.date >= rebalance_dates[0]:
                allocations = self.rebalance(strategy=self)
                if not isinstance(allocations, pd.Series):
                    allocations = pd.Series(allocations, dtype=float)
                if not allocations.empty:
                    self.data["allocations"][self.date] = allocations
                    rebalance_dates = rebalance_dates[1:]

                    # Make trades here
                    target_capials = value * allocations
                    target_shares = target_capials.divide(
                        self.total_prices.loc[self.date]
                    )
                    if self.shares_frac is not None:
                        target_shares = target_shares.round(self.shares_frac)

                    trade_shares = target_shares.subtract(shares, fill_value=0)
                    trade_shares = trade_shares[trade_shares != 0]
                    self.data["trades"][self.date] = trade_shares
                    trade_capitals = trade_shares.multiply(
                        self.total_prices.loc[self.date]
                    )
                    trade_capitals += trade_capitals.multiply(self.commission / 1_000)
                    cash -= trade_capitals.sum()
                    shares = target_shares

            self.data["value"][self.date] = value
            self.data["shares"][self.date] = shares
            self.data["cash"][self.date] = cash
            self.data["weights"][self.date] = weights

    @property
    def analytics(self) -> pd.Series:
        """analytics"""
        return pd.Series(
            data={
                "Start": metrics.to_start(self.value).strftime("%Y-%m-%d"),
                "End": metrics.to_end(self.value).strftime("%Y-%m-%d"),
                "AnnReturn": metrics.to_ann_return(self.value),
                "AnnVolatility": metrics.to_ann_volatility(self.value),
                "SharpeRatio": metrics.to_sharpe_ratio(self.value),
                "SortinoRatio": metrics.to_sortino_ratio(self.value),
                "MaxDrawdown": metrics.to_max_drawdown(self.value),
                "Skewness": metrics.to_skewness(self.value),
                "Kurtosis": metrics.to_kurtosis(self.value),
                "VaR": metrics.to_value_at_risk(self.value),
                "CVaR": metrics.to_conditional_value_at_risk(self.value),
                "TailRatio": metrics.to_tail_ratio(self.value),
                # "Turnover(M)": self.data.trades.resample("M").sum().sum(axis=1).mean(),
            }
        )
