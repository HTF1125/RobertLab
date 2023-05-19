"""ROBERT"""
from typing import Optional, Callable
import pandas as pd
from ..analytics import metrics
from ..ext.clean import clean_weights


class AccountRecords:
    """virtual account records to store account records"""

    def __init__(self) -> None:
        self.value = {}
        self.shares = {}
        self.capitals = {}
        self.prices = {}
        self.weights = {}
        self.allocations = {}


class AccountMetrics:
    """virtual account metrics to store account metrics"""

    def __init__(self, initial_investment: float = 1000.0) -> None:
        self.value: float = initial_investment
        self.date: Optional[pd.Timestamp] = None
        self.shares: pd.Series = pd.Series(dtype=float)
        self.prices: pd.Series = pd.Series(dtype=float)
        self.capitals: pd.Series = pd.Series(dtype=float)
        self.weights: pd.Series = pd.Series(dtype=float)
        self.allocations: pd.Series = pd.Series(dtype=float)


class VirtualAccount:
    """virtual account to store account information"""

    def __init__(self, initial_investment: float = 1000.0) -> None:
        self.metrics = AccountMetrics(initial_investment)
        self.records = AccountRecords()

    ################################################################################
    @property
    def date(self) -> Optional[pd.Timestamp]:
        """account date property"""
        return self.metrics.date

    @date.setter
    def date(self, date: pd.Timestamp) -> None:
        """account date property"""
        self.metrics.date = date

    ################################################################################
    @property
    def value(self) -> float:
        """account value property"""
        return self.metrics.value

    @value.setter
    def value(self, value: float) -> None:
        """account value property"""
        self.metrics.value = value
        self.records.value.update({self.metrics.date: self.value})
        self.weights = self.capitals.divide(self.value)

    ################################################################################
    @property
    def prices(self) -> pd.Series:
        """account value property"""
        return self.metrics.prices

    @prices.setter
    def prices(self, prices: pd.Series) -> None:
        """account value property"""
        if not self.shares.empty:
            self.metrics.prices = prices
            self.records.prices.update({self.metrics.date: self.prices})
            self.capitals = self.shares.multiply(self.prices.fillna(0))

    ################################################################################
    @property
    def shares(self) -> pd.Series:
        """account shares property"""
        return self.metrics.shares

    @shares.setter
    def shares(self, shares: pd.Series) -> None:
        """account shares property"""
        self.metrics.shares = shares
        self.records.shares.update({self.metrics.date: self.shares})

    ################################################################################
    @property
    def capitals(self) -> pd.Series:
        """account capitals property"""
        return self.metrics.capitals

    @capitals.setter
    def capitals(self, capitals: pd.Series) -> None:
        """account capitals property"""
        self.metrics.capitals = capitals
        self.records.capitals.update({self.metrics.date: self.capitals})
        self.value = self.capitals.sum()

    ################################################################################
    @property
    def weights(self) -> pd.Series:
        """account weights property"""
        return self.metrics.weights

    @weights.setter
    def weights(self, weights: pd.Series) -> None:
        """account weights property"""
        self.metrics.weights = weights
        self.records.weights.update({self.metrics.date: self.weights})

    ################################################################################
    @property
    def allocations(self) -> pd.Series:
        """account allocations property"""
        return self.metrics.allocations

    @allocations.setter
    def allocations(self, allocations: pd.Series) -> None:
        """account allocations property"""
        if allocations is not None:
            self.metrics.allocations = allocations
            self.records.allocations.update({self.metrics.date: self.allocations})

    def reset_allocations(self) -> None:
        """reset allocations"""
        self.metrics.allocations = pd.Series(dtype=float)


class Strategy:
    """base strategy"""

    def __init__(
        self,
        prices: pd.DataFrame,
        rebalance: Callable,
        frequency: str = "M",
        initial_investment: float = 1000.0,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> None:
        self.account: VirtualAccount = VirtualAccount(
            initial_investment=initial_investment
        )
        self.prices: pd.DataFrame = prices.ffill()
        self.rebalance: Callable = rebalance
        self.frequency: str = frequency
        self.simulate(
            start=start or str(self.prices.index[0]),
            end=end or str(self.prices.index[-1]),
        )

    @property
    def date(self) -> Optional[pd.Timestamp]:
        """date property"""
        return self.account.date

    @date.setter
    def date(self, date: pd.Timestamp) -> None:
        """date property"""
        self.account.date = date

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
    def allocations(self) -> pd.DataFrame:
        """strategy allocations time-series"""
        return pd.DataFrame(
            data=self.account.records.allocations, dtype=float
        ).T.sort_index()

    ################################################################################

    @property
    def reb_prices(self) -> pd.DataFrame:
        """rebalancing prices"""
        if self.date is None:
            return pd.DataFrame()
        return self.prices[self.prices.index < self.date].dropna(how="all", axis=1)

    def make_allocation(self) -> pd.Series:
        """wrapper"""
        allocations = self.rebalance(strategy=self)
        if not isinstance(allocations, pd.Series):
            return pd.Series(allocations, dtype=float)
        return allocations

    def simulate(self, start: str, end: str) -> None:
        """simulate strategy"""

        reb_dates = pd.date_range(start=start, end=end, freq=self.frequency)
        rebalance = True
        for self.date in pd.date_range(start=start, end=end, freq="D"):
            if self.date not in self.prices.index:
                continue
            self.account.prices = self.prices.loc[self.date]
            if rebalance:
                self.account.allocations = clean_weights(
                    weights=self.make_allocation(),
                    num_decimal=4,
                )
                if not self.account.allocations.empty:
                    rebalance = False
            if self.date in self.prices.index and not self.account.allocations.empty:
                self.account.weights = self.account.allocations
                self.account.capitals = self.account.value * self.account.weights
                self.account.shares = self.account.capitals.divide(
                    self.prices.loc[self.date].dropna()
                )
                self.account.reset_allocations()

            if not rebalance:
                rebalance = self.date in reb_dates

    @property
    def analytics(self) -> pd.Series:
        """analytics"""
        return pd.Series(
            data={
                "AnnReturn": metrics.to_ann_return(self.value),
                "AnnVolatility": metrics.to_ann_volatility(self.value),
                "SharpeRatio": metrics.to_sharpe_ratio(self.value),
                "Skewness": metrics.to_skewness(self.value),
                "Kurtosis": metrics.to_kurtosis(self.value),
                "VaR": metrics.to_value_at_risk(self.value),
                "CVaR": metrics.to_conditional_value_at_risk(self.value),
            }
        )
