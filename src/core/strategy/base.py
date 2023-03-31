import sys
from typing import Optional
import numpy as np
import pandas as pd
from .account import VirtualAccount
from ..analytics import metrics
from ..analytics import utils
from ..portfolio import optimizer


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
    def reb_prices(self) -> pd.DataFrame:
        """get rebalancing prices"""
        return self.prices.loc[: self.date].dropna(thresh=self.min_num_period, axis=1)

    @property
    def date(self) -> pd.Timestamp:
        """date property"""
        return self.account.date

    @date.setter
    def date(self, date: pd.Timestamp) -> None:
        """date property"""
        self.account.date = date
        if date in self.prices.index:
            self.update_book()

    @staticmethod
    def clean_weights(weights: pd.Series, num_decimal: int = 4, num_rebase: int = 100) -> pd.Series:
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

    def sell(self, asset: str, share: int) -> None:
        
        pass



    @staticmethod
    def _update_book(
        value: float,
        shares: Optional[pd.Series],
        capitals: Optional[pd.Series],
        prices: Optional[pd.Series],
        allocations: Optional[pd.Series],
    ):
        if shares is not None:
            new_capitals = shares.multiply(prices)
            if capitals is not None:
                profits = new_capitals.subtract(capitals)
                new_value = value + profits.sum()
            else:
                new_value = value
            new_weights = new_capitals.multiply(1 / new_value)
        else:
            new_value = value
        new_shares = shares
        if allocations is not None:
            if new_weights is not None:
                trades = allocations.subtract(new_weights, fill_value=0).abs()
            else:
                trades = allocations.abs()
            new_weights = allocations
            new_capitals = new_weights.multiply(new_value)
            new_shares = capitals.divide(prices)
        new_allocations = None
        return {
            "value": new_value,
            "shares": new_shares,
            "capitals": new_capitals,
            "weights": new_weights,
            "allocations": new_allocations,
            "trades": trades,
            "profits": profits,
        }

    def update_book(self) -> None:
        if self.account.shares is not None:
            capitals = self.account.shares * self.prices.loc[self.date].dropna()
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
            self.account.shares = self.account.capitals.divide(self.prices.loc[self.date].dropna())
            self.account.allocations = None

    @property
    def value(self) -> pd.Series:
        val = pd.Series(self.account.records.value).sort_index()
        val.name = "value"
        return val

    def rebalance(self, **kwargs) -> pd.Series:
        """Default rebalancing method"""
        asset = self.reb_prices.iloc[-1].dropna().index
        uniform_weight = np.ones(len(asset))
        uniform_weight /= uniform_weight.sum()
        weight = pd.Series(index=asset, data=uniform_weight)
        return weight

    def simulate(self, start: ... = None, end: ... = None, **kwargs) -> "Strategy":
        """_summary_

        Args:
            start (None, optional): _description_. Defaults to None.
            end (None, optional): _description_. Defaults to None.

        Returns:
            Strategy: _description_
        """
        start = start or self.prices.index[self.min_num_period]
        end = end or self.prices.index[-1]
        reb_dates = pd.date_range(start=start, end=end, freq=self.reb_frequency)
        bar_dates = pd.date_range(start=start, end=end, freq=self.bar_frequency)
        num_bar_dates = len(bar_dates)
        make_rebalance = True
        for idx, self.date in enumerate(bar_dates, 1):
            self.terminal_progress(
                idx, num_bar_dates, "simulate", f"{self.account.value:.2f}"
            )
            if self.date in reb_dates:
                make_rebalance = True
            if make_rebalance:
                if self.reb_prices.empty:
                    continue
                self.account.allocations = self.rebalance(**kwargs)
                if self.account.allocations is not None:
                    self.account.allocations = self.clean_weights(
                        weights=self.account.allocations, num_decimal=4
                    )
                    make_rebalance = False
        return self

    def analytics(self) -> pd.Series:
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

        covariance_matrix = utils.to_covariance_matrix(
            prices=self.prices.loc[: self.date], halflife=63
        )

        opt = optimizer.Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        weights = opt.hierarchical_equal_risk_contribution()
        return weights


class HierarchicalRiskParity(Strategy):
    def rebalance(self, **kwargs):

        covariance_matrix = utils.to_covariance_matrix(
            prices=self.prices.loc[: self.date].iloc[-252:]
        )

        opt = optimizer.Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        return opt.hierarchical_risk_parity()


class RiskParity(Strategy):
    def rebalance(self, **kwargs):

        covariance_matrix = utils.to_covariance_matrix(
            prices=self.prices.loc[: self.date],
            halflife=21,
        )

        opt = optimizer.Optimizer(covariance_matrix=covariance_matrix, **kwargs)
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

        covariance_matrix = utils.to_covariance_matrix(
            prices=self.prices.loc[: self.date].iloc[-252:]
        )
        opt = optimizer.Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        return opt.inverse_variance()


class TargetVol(Strategy):
    def rebalance(self, **kwargs):

        covariance_matrix = utils.to_covariance_matrix(
            prices=self.prices.loc[: self.date].iloc[-252:]
        )
        opt = optimizer.Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        return opt.minimized_volatility()


class Momentum(Strategy):
    def rebalance(self, **kwargs) -> pd.Series:

        prices = self.prices.loc[: self.date]
        momentum_1y = prices.iloc[-1] / prices.iloc[-21]

        momentum_1y = momentum_1y.dropna().nsmallest(6)

        prices = prices[momentum_1y.index]
        covariance_matrix = utils.to_covariance_matrix(prices=prices, halflife=21)
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
