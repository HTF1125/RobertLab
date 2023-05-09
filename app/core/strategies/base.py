"""ROBERT"""
from typing import Optional, Callable
import pandas as pd
from .account import VirtualAccount
from ..analytics import metrics
from ..ext.progress import terminal_progress
from ..ext.clean import clean_weights


class Strategy:
    """base strategy"""

    def __init__(
        self,
        prices: pd.DataFrame,
        rebalance: Callable,
        frequency: str = "M",
        initial_investment: float = 1000.0,
        min_periods: int = 2,
        min_assets: int = 2,
    ) -> None:
        self.account = VirtualAccount(initial_investment=initial_investment)
        self.prices = prices.ffill()
        self.rebalance = rebalance
        self.frequency = frequency
        self.min_periods = min_periods
        self.min_assets = min_assets

    @property
    def date(self) -> Optional[pd.Timestamp]:
        """date property"""
        return self.account.date

    @date.setter
    def date(self, date: pd.Timestamp) -> None:
        """date property"""
        self.account.date = date
        self.account.prices = self.prices.loc[self.date]

    @property
    def reb_prices(self) -> Optional[pd.DataFrame]:
        """rebalancing prices"""
        reb_prices = self.prices.loc[: self.date].copy()
        reb_prices.dropna(thresh=self.min_periods, axis=1, inplace=True)
        if len(reb_prices) < self.min_assets:
            return None
        if len(reb_prices.columns) < self.min_assets:
            return None
        return reb_prices

    ################################################################################

    @property
    def value(self) -> pd.Series:
        """strategy value time-series"""
        val = pd.Series(
            data=self.account.records.value, name="value", dtype=float
        ).sort_index()
        return val

    @property
    def weights(self) -> pd.DataFrame:
        """strategy weights time-series"""
        return pd.DataFrame(
            data=self.account.records.weights, dtype=float
        ).T.sort_index()

    @property
    def capitals(self) -> pd.DataFrame:
        """strategy capitals time-series"""
        return pd.DataFrame(
            data=self.account.records.capitals, dtype=float
        ).T.sort_index()

    @property
    def profits(self) -> pd.DataFrame:
        """strategy profits time-series"""
        return pd.DataFrame(
            data=self.account.records.profits, dtype=float
        ).T.sort_index()

    @property
    def allocations(self) -> pd.DataFrame:
        """strategy allocations time-series"""
        return pd.DataFrame(
            data=self.account.records.allocations, dtype=float
        ).T.sort_index()

    ################################################################################

    def simulate(self, *args, **kwargs) -> "Strategy":
        """"""
        reb_dates = [
            self.prices.loc[date:].index[0]
            for date in pd.date_range(
                start=str(self.prices.index[0]),
                end=str(self.prices.index[-1]),
                freq=self.frequency,
            )
        ]
        total_bar = len(self.prices)
        for idx, self.date in enumerate(self.prices.index, 1):
            terminal_progress(
                current_bar=idx,
                total_bar=total_bar,
                prefix="simulate",
                suffix=f"{self.date:%Y-%m-%d} - {self.account.value:.2f}",
            )
            if not self.account.allocations.empty:
                self.account.trades = self.account.allocations.subtract(
                    self.account.weights, fill_value=0
                ).abs()
                self.account.weights = self.account.allocations
                self.account.capitals = self.account.value * self.account.weights
                self.account.shares = self.account.capitals.divide(
                    self.prices.loc[self.date].dropna()
                )
                self.account.reset_allocations()
            if self.date in reb_dates or self.account.shares.empty:
                reb_prices = self.reb_prices
                if reb_prices is not None:
                    try:
                        allocations = self.rebalance(prices=reb_prices, *args, **kwargs)
                        self.account.allocations = clean_weights(
                            weights=allocations, num_decimal=4
                        )
                    except:
                        self.account.reset_allocations()

        return self

    def analytics(self) -> pd.DataFrame:
        """analytics"""
        return pd.concat(
            [
                metrics.to_ann_return(self.value.to_frame()),
                metrics.to_ann_volatility(self.value.to_frame()),
                metrics.to_sharpe_ratio(self.value.to_frame()),
            ]
        )
