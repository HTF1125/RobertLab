import sys
from typing import Optional
import numpy as np
import pandas as pd
from ..analytics import metrics
from ..analytics import estimators
from ..portfolios import optimizer
from ..ext.progress import terminal_progress


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
        self.value: float = initial_investment
        self.date: Optional[pd.Timestamp] = None
        self.shares: pd.Series = pd.Series(dtype=float)
        self.prices: pd.Series = pd.Series(dtype=float)
        self.capitals: pd.Series = pd.Series(dtype=float)
        self.weights: pd.Series = pd.Series(dtype=float)
        self.profits: pd.Series = pd.Series(dtype=float)
        self.trades: pd.Series = pd.Series(dtype=float)
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

    ################################################################################
    @property
    def prices(self) -> pd.Series:
        """account value property"""
        return self.metrics.prices

    @prices.setter
    def prices(self, prices: pd.Series) -> None:
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
        self.weights = self.capitals.divide(self.value)

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
        self.metrics.allocations = allocations
        self.records.allocations.update({self.metrics.date: self.allocations})

    def reset_allocations(self) -> None:
        """reset allocations"""
        self.metrics.allocations = pd.Series(dtype=float)

    ################################################################################
    @property
    def trades(self) -> pd.Series:
        """account trades property"""
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
        return self.metrics.profits

    @profits.setter
    def profits(self, profits: pd.Series) -> None:
        """account profits property"""
        self.metrics.profits = profits
        self.records.profits.update({self.metrics.date: self.profits})


class Strategy:
    """base strategy"""

    def __init__(
        self,
        prices: pd.DataFrame,
        frequency: str = "M",
        initial_investment: float = 1000.0,
        min_periods: int = 2,
        min_assets: int = 2,
    ) -> None:
        self.account = VirtualAccount(initial_investment=initial_investment)
        self.prices = prices.ffill()
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
    def reb_prices(self) -> pd.DataFrame:
        """rebalancing prices"""
        return self.prices.loc[: self.date].dropna(thresh=self.min_periods, axis=1)

    ################################################################################

    @property
    def value(self) -> pd.Series:
        val = pd.Series(self.account.records.value, name="value").sort_index()
        return val

    @property
    def weights(self) -> pd.DataFrame:
        return pd.DataFrame(self.account.records.weights).T.sort_index()

    @property
    def capitals(self) -> pd.DataFrame:
        return pd.DataFrame(self.account.records.capitals).T.sort_index()

    @property
    def profits(self) -> pd.DataFrame:
        return pd.DataFrame(self.account.records.profits).T.sort_index()

    @property
    def allocations(self) -> pd.DataFrame:
        return pd.DataFrame(self.account.records.allocations).T.sort_index()

    ################################################################################

    @staticmethod
    def clean_weights(
        weights: pd.Series, num_decimal: int = 4, num_rebase: int = 100
    ) -> pd.Series:
        """Clean weights based on the number decimals and maintain the total of weights.

        Args:
            weights (pd.Series): asset weights.
            decimals (int, optional): number of round decimals. Defaults to 4.

        Returns:
            pd.Series: cleaned asset weights.
        """
        # clip weight values by minimum and maximum.
        tot_weight = weights.sum().round(num_decimal)
        weights = weights.round(decimals=num_decimal)
        # repeat round and weight calculation.
        for _ in range(num_rebase):
            weights = weights / weights.sum() * tot_weight
            weights = weights.round(decimals=num_decimal)
            if weights.sum() == tot_weight:
                return weights
        # if residual remains after repeated rounding.
        # allocate the the residual weight on the max weight.
        residual = tot_weight - weights.sum()
        # !!! Error may occur when there are two max weights???
        weights[np.argmax(weights)] += np.round(residual, decimals=num_decimal)
        return weights

    def rebalance(self) -> pd.Series:
        """Default rebalancing method"""
        asset = self.prices.loc[: self.date].iloc[-1].dropna().index
        uniform_weight = np.ones(len(asset))
        uniform_weight /= uniform_weight.sum()
        weight = pd.Series(index=asset, data=uniform_weight)
        return weight

    def simulate(
        self, start: Optional[str] = None, end: Optional[str] = None
    ) -> "Strategy":
        """_summary_

        Args:
            start (None, optional): start date of simulation. Defaults to None.
            end (None, optional): end date of simulation. Defaults to None.

        Returns:
            Strategy: _description_
        """
        start = start or self.prices.index[0]
        end = end or self.prices.index[-1]

        reb_dates = [
            self.prices.loc[date:].index[0]
            for date in pd.date_range(start=start, end=end, freq=self.frequency)
        ]
        make_rebalance = True
        total_bar = len(self.prices.loc[start:end].index)
        for idx, self.date in enumerate(self.prices.loc[start:end].index, 1):
            terminal_progress(
                current_bar=idx,
                total_bar=total_bar,
                prefix="simulate",
                suffix=f"{self.date} - {self.account.value:.2f}",
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
            if len(self.reb_prices) <= self.min_periods:
                continue
            if self.date in reb_dates:
                make_rebalance = True
            if make_rebalance:
                self.account.allocations = self.rebalance()
                if self.account.allocations is None:
                    self.account.reset_allocations()
                    continue
                if not self.account.allocations.empty:
                    self.account.allocations = self.clean_weights(
                        weights=self.account.allocations, num_decimal=4
                    )
                    make_rebalance = False

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
        covariance_matrix = estimators.to_covariance_matrix(
            prices=self.reb_prices, halflife=63
        )
        correlation_matrix = estimators.to_correlation_matrix(
            prices=self.reb_prices, halflife=63
        )

        opt = optimizer.Optimizer(
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
            **kwargs,
        )
        weights = opt.hierarchical_equal_risk_contribution()
        return weights


class HierarchicalRiskParity(Strategy):
    def rebalance(self, **kwargs):

        covariance_matrix = estimators.to_covariance_matrix(
            prices=self.reb_prices, halflife=63
        )
        correlation_matrix = estimators.to_correlation_matrix(
            prices=self.reb_prices, halflife=63
        )

        opt = optimizer.Optimizer(
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
            **kwargs,
        )
        return opt.hierarchical_risk_parity()


class RiskParity(Strategy):
    def rebalance(self, **kwargs):
        covariance_matrix = estimators.to_covariance_matrix(
            prices=self.reb_prices, halflife=63
        )
        correlation_matrix = estimators.to_correlation_matrix(
            prices=self.reb_prices, halflife=63
        )

        opt = optimizer.Optimizer(
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
            **kwargs,
        )
        return opt.risk_parity()


class MaxSharpe(Strategy):
    def rebalance(self, **kwargs):
        prices = self.prices.loc[: self.date].iloc[-252:]
        cov = prices.pct_change().fillna(0).cov() * (252**0.5)
        er = prices.pct_change().fillna(0).mean() * (252)
        opt = optimizer.Optimizer(expected_returns=er, covariance_matrix=cov, **kwargs)
        return opt.maximized_sharpe_ratio()


class InverseVariance(Strategy):
    def rebalance(self, **kwargs):
        covariance_matrix = estimators.to_covariance_matrix(
            prices=self.prices.loc[: self.date].iloc[-252:]
        )
        opt = optimizer.Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        return opt.inverse_variance()


class TargetVol(Strategy):
    def rebalance(self, **kwargs):
        covariance_matrix = estimators.to_covariance_matrix(
            prices=self.prices.loc[: self.date].iloc[-252:]
        )
        opt = optimizer.Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        return opt.minimized_volatility()


class Momentum(Strategy):
    def rebalance(self, **kwargs) -> Optional[pd.Series]:
        prices = self.prices.loc[: self.date]
        momentum_1y = prices.iloc[-1] / prices.iloc[-21]

        momentum_1y = momentum_1y.dropna().nsmallest(6)

        prices = prices[momentum_1y.index]
        covariance_matrix = estimators.to_covariance_matrix(prices=prices, halflife=21)
        opt = optimizer.Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        return opt.hierarchical_equal_risk_contribution()


class MeanReversion(Strategy):
    """
    What is Mean Reversion?
        According to Investopedia, mean reversion, or reversion to the mean, is
        a theory used in finance (rooted in a concept well known as regression
        towards the mean) that suggests that asset price volatility and
        historical returns eventually will revert to the long-run mean or
        average level of the entire dataset. Mean is the average price and
        reversion means to return to, so mean reversion means “return to the
        average price”.

        While an assets price tends to revert to the average over time, this
        does not always mean or guarantee that the price will go back to the
        mean, nor does it mean that the price will rise to the mean.

    What Is A Mean Reversion Trading Strategy ?
        A mean reversion trading strategy is a trading strategy that focuses on
        when a security moves too far away from some average. The theory is that
        the price will move back toward that average at some point in time.
        There are many different ways to look at this strategy, for example by
        using linear regression, RSI, Bollinger Bands, standard deviation,
        moving averages etc. The question is how far from the average / mean is
        too far ?
    """

    def rebalance(self, **kwargs) -> pd.Series:
        return super().rebalance(**kwargs)
