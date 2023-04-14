""""""


import pandas as pd


class AccountRecords:
    """virtual account records to store account records"""

    def __init__(self) -> None:
        self.value = {}
        self.shares = {}
        self.capitals = {}
        self.prices = {}
        self.profits = {}
        self.trades = {}
        self.allocations = {}
        self.weights = {}


class AccountMetrics:
    """virtual account metrics to store account metrics"""

    def __init__(self, initial_investment: float = 1000.0) -> None:
        self.date = None
        self.value = initial_investment
        self.shares = None
        self.prices = None
        self.capitals = None
        self.weights = None
        self.profits = None
        self.trades = None
        self.allocations = None


class VirtualAccount:
    """virtual account to store account information"""

    def __init__(self, initial_investment: float = 1000.0) -> None:
        self.metrics = AccountMetrics(initial_investment)
        self.records = AccountRecords()

    ################################################################################
    @property
    def date(self) -> float:
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

    ################################################################################
    @property
    def prices(self) -> pd.Series:
        """account value property"""
        if self.metrics.prices is None:
            return pd.Series(dtype=float)
        return self.metrics.prices

    @prices.setter
    def prices(self, prices: float) -> None:
        """account value property"""
        self.metrics.prices = prices
        self.records.prices.update({self.metrics.date: self.prices})
        if not self.shares.empty:
            capitals = self.shares.multiply(self.prices.fillna(0))
            self.profits = capitals.subtract(self.capitals)
            self.capitals = capitals

    ################################################################################
    @property
    def shares(self) -> pd.Series:
        """account shares property"""
        if self.metrics.shares is None:
            return pd.Series(dtype=float)
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
        if self.metrics.capitals is None:
            return pd.Series(dtype=float)
        return self.metrics.capitals

    @capitals.setter
    def capitals(self, capitals: pd.Series) -> None:
        """account capitals property"""
        self.metrics.capitals = capitals
        self.records.capitals.update({self.metrics.date: self.capitals})
        self.value = self.capitals.sum()
        self.weights = self.capitals.divide(self.value)

    ################################################################################
    @property
    def weights(self) -> pd.Series:
        """account weights property"""
        if self.metrics.weights is None:
            return pd.Series(dtype=float)
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
        if self.metrics.allocations is None:
            return pd.Series(dtype=float)
        return self.metrics.allocations

    @allocations.setter
    def allocations(self, allocations: pd.Series) -> None:
        """account allocations property"""
        self.metrics.allocations = allocations
        if self.metrics.allocations is not None:
            self.records.allocations.update({self.metrics.date: self.allocations})

    ################################################################################
    @property
    def trades(self) -> pd.Series:
        """account trades property"""
        if self.metrics.trades is None:
            return pd.Series(dtype=float)
        return self.metrics.trades

    @trades.setter
    def trades(self, trades: pd.Series) -> None:
        """account trades property"""
        self.metrics.trades = trades
        self.records.trades.update({self.metrics.date: self.trades})

    ################################################################################
    @property
    def profits(self) -> pd.Series:
        """account profits property"""
        if self.metrics.profits is None:
            return pd.Series(dtype=float)
        return self.metrics.profits

    @profits.setter
    def profits(self, profits: pd.Series) -> None:
        """account profits property"""
        self.metrics.profits = profits
        self.records.profits.update({self.metrics.date: self.profits})
