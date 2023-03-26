import sys
from typing import Dict
import numpy as np
import pandas as pd
from .. import metrics


class AccountRecord:
    def __init__(self) -> None:

        self.value: Dict = {}
        self.shares: Dict = {}
        self.capitals: Dict = {}
        self.profits: Dict = {}
        self.trades: Dict = {}
        self.allocations: Dict = {}
        self.weights: Dict = {}


class VirtualAccount:
    def __init__(self, initial_investment: float = 1000.0) -> None:
        self.metrics: Dict = {}
        self.records = AccountRecord()
        self.initial_investment = initial_investment

    @property
    def value(self) -> pd.Series:
        return self.metrics.get("value", self.initial_investment)

    @value.setter
    def value(self, value: pd.Series) -> None:
        if value is not None:
            self.metrics.update({"value": value})
            self.records.value.update({self.date: self.value})

    @property
    def allocations(self) -> pd.Series:
        return self.metrics.get("allocations", None)

    @allocations.setter
    def allocations(self, allocations: pd.Series) -> None:
        if allocations is not None:
            self.metrics.update({"allocations": allocations})
            self.records.allocations.update({self.date: self.allocations})

    @property
    def weights(self) -> pd.Series:
        return self.metrics.get("weights", None)

    @weights.setter
    def weights(self, weights: pd.Series) -> None:
        if weights is not None:
            self.metrics.update({"weights": weights})
            self.records.weights.update({self.date: self.weights})

    @property
    def shares(self) -> pd.Series:
        return self.metrics.get("shares", None)

    @shares.setter
    def shares(self, shares: pd.Series) -> None:
        if shares is not None:
            self.metrics.update({"shares": shares})
            self.records.shares.update({self.date: self.shares})

    @property
    def capitals(self) -> pd.Series:
        return self.metrics.get("capitals", None)

    @capitals.setter
    def capitals(self, capitals: pd.Series) -> None:
        if capitals is not None:
            self.metrics.update({"capitals": capitals})
            self.records.capitals.update({self.date: self.capitals})

    @property
    def profits(self) -> pd.Series:
        return self.metrics.get("profits", None)

    @profits.setter
    def profits(self, profits: pd.Series) -> None:
        if profits is not None:
            self.metrics.update({"profits": profits})
            self.records.profits.update({self.date: self.profits})

    @property
    def trades(self) -> pd.Series:
        return self.metrics.get("trades", None)

    @trades.setter
    def trades(self, trades: pd.Series) -> None:
        if trades is not None:
            self.metrics.update({"trades": trades})
            self.records.trades.update({self.date: self.trades})


class Strategy:
    """base strategy"""

    def __init__(
        self,
        prices: pd.DataFrame,
        reb_frequency: str = "M",
        bar_frequency: str = "D",
        initial_investment: float = 1000.0,
        min_num_asset: int = 2,
        min_num_period: int = 252,
    ) -> None:
        self.account = VirtualAccount(initial_investment=initial_investment)

        self.prices = prices.ffill()
        self.date = prices.index[0]
        self.reb_frequency = reb_frequency
        self.bar_frequency = bar_frequency
        self.min_num_asset = min_num_asset
        self.min_num_period = min_num_period

    @property
    def date(self) -> pd.Timestamp:
        return self.account.date

    @date.setter
    def date(self, date: pd.Timestamp) -> None:

        self.account.date = date
        if date in self.prices.index:
            self.update_book()

    @property
    def price(self) -> pd.Series:
        return self.prices.loc[: self.date].iloc[-1].dropna()

    def update_book(self) -> None:
        if self.account.shares is not None:
            capitals = self.account.shares * self.price
            if self.account.capitals is not None:
                self.account.profits = capitals.dropna() - self.account.capitals
                self.account.value += self.account.profits.sum()
            self.account.capitals = capitals
            self.account.weights = self.account.capitals.divide(self.account.value)
        if self.account.allocations is not None:
            if self.account.weights is not None:
                self.account.trades = self.account.allocations.subtract(
                    self.account.weights, fill_value=0
                ).abs()
            else:
                self.account.trades = self.account.allocations
            self.account.weights = self.account.allocations
            self.account.capitals = self.account.value * self.account.weights
            self.account.shares = self.account.capitals.divide(self.price)
            self.account.allocations = None

    @property
    def value(self) -> pd.Series:
        val = pd.Series(self.account.records.value)
        val.name = "value"
        return val

    def rebalance(self, **kwargs) -> pd.Series:
        """Default rebalancing method"""
        asset = self.price.index
        uniform_weight = np.ones(len(asset))
        uniform_weight /= uniform_weight.sum()
        weight = pd.Series(index=asset, data=uniform_weight)
        return weight

    def simulate(self, start: ... = None, end: ... = None, **kwargs) -> "Strategy":

        make_rebalance = True
        start = start or self.prices.index[self.min_num_period]
        end = end or self.prices.index[-1]
        reb_dates = pd.date_range(start=start, end=end, freq=self.reb_frequency)

        bar_dates = pd.date_range(start=start, end=end, freq=self.bar_frequency)
        num_bar_dates = len(bar_dates)
        for idx, self.date in enumerate(bar_dates, 1):

            self.terminal_progress(
                idx, num_bar_dates, "simulate", f"{self.date} - {self.account.value}"
            )
            if self.date in reb_dates:
                make_rebalance = True
            if make_rebalance:
                price_slice = (
                    self.prices.loc[: self.date]
                    .dropna(thresh=self.min_num_period, axis=1)
                    .dropna(thresh=self.min_num_asset, axis=0)
                )
                if price_slice.empty:
                    continue
                self.account.allocations = self.rebalance(**kwargs)
                if self.account.allocations is not None:
                    make_rebalance = False
        return self

    def analytics(self) -> pd.Series:

        return pd.concat(
            [
                metrics.to_ann_returns(self.value.to_frame()),
                metrics.to_ann_volatilites(self.value.to_frame()),
                metrics.to_sharpe_ratios(self.value.to_frame()),
            ]
        )

    @staticmethod
    def terminal_progress(
        current_bar: int,
        total_bar: int,
        prefix: str = "",
        suffix: str = "",
        bar_length: int = 50,
    ) -> None:
        # pylint: disable=expression-not-assigned
        """
        Calls in a loop to create a terminal progress bar.

        Args:
            current_bar (int): Current iteration.
            total_bar (int): Total iteration.
            prefix (str, optional): Prefix string. Defaults to ''.
            suffix (str, optional): Suffix string. Defaults to ''.
            bar_length (int, optional): Character length of the bar.
                Defaults to 50.

        References:
            https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
        """
        # Calculate the percent completed.
        percents = current_bar / float(total_bar)
        # Calculate the length of bar.
        filled_length = int(round(bar_length * current_bar / float(total_bar)))
        # Fill the bar.
        block = "█" * filled_length + "-" * (bar_length - filled_length)
        # Print new line.
        sys.stdout.write(f"\r{prefix} |{block}| {percents:.2%} {suffix}")

        if current_bar == total_bar:
            sys.stdout.write("\n")
        sys.stdout.flush()


class HierarchicalEqualRiskContribution(Strategy):
    def rebalance(self, **kwargs):
        from ..optimizer import Optimizer
        from ..metrics import to_covariance_matrix

        covariance_matrix = to_covariance_matrix(
            prices=self.prices.loc[: self.date].iloc[-21:]
        )

        opt = Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        weights = opt.hierarchical_equal_risk_contribution()
        return weights


class HierarchicalRiskParity(Strategy):
    def rebalance(self, **kwargs):
        from ..optimizer import Optimizer
        from ..metrics import to_covariance_matrix

        covariance_matrix = to_covariance_matrix(
            prices=self.prices.loc[: self.date].iloc[-252:]
        )

        opt = Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        return opt.hierarchical_risk_parity()


class RiskParity(Strategy):
    def rebalance(self, **kwargs):
        from ..optimizer import Optimizer
        from ..metrics import to_covariance_matrix

        covariance_matrix = to_covariance_matrix(
            prices=self.prices.loc[: self.date],
            halflife=21,
        )

        opt = Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        return opt.risk_parity()


class MaxSharpe(Strategy):
    def rebalance(self, **kwargs):
        from ..optimizer import Optimizer

        prices = self.prices.loc[: self.date].iloc[-252:]
        cov = prices.pct_change().fillna(0).cov() * (252**0.5)
        er = prices.pct_change().fillna(0).mean() * (252)
        opt = Optimizer(expected_returns=er, covariance_matrix=cov, **kwargs)
        return opt.maximized_sharpe_ratio()


class InverseVariance(Strategy):
    def rebalance(self, **kwargs):
        from ..optimizer import Optimizer
        from ..metrics import to_covariance_matrix

        covariance_matrix = to_covariance_matrix(
            prices=self.prices.loc[: self.date].iloc[-252:]
        )
        opt = Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        return opt.inverse_variance()


class TargetVol(Strategy):
    def rebalance(self, **kwargs):
        from ..optimizer import Optimizer
        from ..metrics import to_covariance_matrix

        covariance_matrix = to_covariance_matrix(
            prices=self.prices.loc[: self.date].iloc[-252:]
        )
        opt = Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        return opt.minimized_volatility()


class Momentum(Strategy):
    def rebalance(self, **kwargs) -> pd.Series:
        from ..optimizer import Optimizer
        from ..metrics import to_covariance_matrix

        prices = self.prices.loc[: self.date]
        momentum_1y = prices.iloc[-1] / prices.iloc[-21]

        momentum_1y = momentum_1y.dropna().nsmallest(6)

        prices = prices[momentum_1y.index]
        covariance_matrix = to_covariance_matrix(prices=prices, halflife=21)
        opt = Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        return opt.hierarchical_equal_risk_contribution()
