"""ROBERT"""
from typing import Optional, Callable, Iterator
import pandas as pd
from ..ext.store import DataStore
from .. import metrics


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
        """
        Initialize a Strategy object.

        Args:
            prices (pd.DataFrame): Price data for the strategy.
            rebalance (Callable): Callable object for rebalancing the strategy.
            frequency (str, optional): Frequency of rebalancing. Defaults to "M".
            start (str, optional): Start date for the simulation. Defaults to None.
            end (str, optional): End date for the simulation. Defaults to None.
            initial_investment (float, optional): Initial investment amount. Defaults to 10_000.0.
            commission (int, optional): Commission amount for trades. Defaults to 10.
            shares_frac (int, optional): Number of decimal places for rounding shares. Defaults to None.
            prices_bm (pd.Series, optional): Benchmark price data. Defaults to None.
        """
        self.total_prices: pd.DataFrame = prices.ffill()
        self.commission = commission
        self.rebalance: Callable = rebalance
        self.shares_frac = shares_frac
        self.initial_investment = initial_investment
        self.data = DataStore()
        self.start = start or str(self.total_prices.index[0])
        self.end = end or str(self.total_prices.index[-1])
        self.freq = frequency
        self.date: pd.Timestamp = pd.Timestamp(
            str(self.total_prices.loc[: self.start].index[-1])
        )
        self.simulate()

    ################################################################################

    @property
    def prices(self) -> pd.DataFrame:
        """
        Get the price data for the strategy.

        Returns:
            pd.DataFrame: Price data.
        """
        if self.date is None:
            return pd.DataFrame()
        return self.total_prices.loc[: self.date].dropna(how="all", axis=1)

    @property
    def value(self) -> pd.Series:
        """
        Get the value of the strategy.

        Returns:
            pd.Series: Strategy value.
        """
        return pd.Series(self.data.get("value"))

    @property
    def cash(self) -> pd.Series:
        """
        Get the cash holdings of the strategy.

        Returns:
            pd.Series: Cash holdings.
        """
        return pd.Series(self.data.get("cash"))

    @property
    def allocations(self) -> pd.DataFrame:
        """
        Get the asset allocations of the strategy.

        Returns:
            pd.DataFrame: Asset allocations.
        """
        return pd.DataFrame(self.data.get("allocations")).T

    @property
    def weights(self) -> pd.DataFrame:
        """
        Get the asset weights of the strategy.

        Returns:
            pd.DataFrame: Asset weights.
        """
        return pd.DataFrame(self.data.get("weights")).T

    ################################################################################
    def generate_rebalance_dates(self) -> Iterator[pd.Timestamp]:
        """
        Generate rebalance dates between the given start and end dates with the specified frequency.

        Yields:
            Iterator[pd.Timestamp]: Iterator that yields rebalance dates.
        """
        for rebalance_date in pd.date_range(
            start=self.start, end=self.end, freq=self.freq
        ):
            yield rebalance_date

    def simulate(self) -> None:
        """
        Simulate the strategy.
        """
        cash = self.initial_investment
        shares = pd.Series(dtype=float)
        # generate rebalance dates
        rebalance_dates = self.generate_rebalance_dates()
        rebalance_date = next(rebalance_dates)
        for date in self.total_prices.loc[self.start : self.end].index:
            capitals = shares.multiply(self.total_prices.loc[date]).dropna()
            value = sum(capitals) + cash
            weights = capitals.divide(value)
            if date > rebalance_date or value == cash:
                allocations = self.rebalance(strategy=self)
                if isinstance(allocations, pd.Series):
                    self.data["allocations"][self.date] = allocations
                    target_capials = value * allocations
                    target_shares = target_capials.divide(self.total_prices.loc[date])
                    if self.shares_frac is not None:
                        target_shares = target_shares.round(self.shares_frac)
                    trade_shares = target_shares.subtract(shares, fill_value=0)
                    trade_shares = trade_shares[trade_shares != 0]
                    self.data["trades"][date] = trade_shares
                    trade_capitals = trade_shares.multiply(self.total_prices.loc[date])
                    trade_capitals += trade_capitals.multiply(self.commission / 1_000)
                    cash -= trade_capitals.sum()
                    shares = target_shares
                try:
                    if date > rebalance_date:
                        rebalance_date = next(rebalance_dates)
                except StopIteration:
                    rebalance_date = self.total_prices.loc[: self.end].index[-2]

            self.date = date
            self.data["value"][self.date] = value
            self.data["shares"][self.date] = shares
            self.data["cash"][self.date] = cash
            self.data["weights"][self.date] = weights

    def information_coefficient(self) -> float:
        from scipy import stats

        chg = self.allocations - self.allocations.mean()
        fwd = self.total_prices.loc[self.allocations.index].pct_change().shift(-1)
        joined = pd.concat([chg.stack(), fwd.stack()], axis=1).dropna()
        return stats.spearmanr(joined.iloc[:, 0], joined.iloc[:, 1])[0]

    @property
    def analytics(self) -> pd.Series:
        """
        Get the analytics of the strategy.

        Returns:
            pd.Series: Strategy analytics.
        """
        return pd.Series(
            data={
                "AnnReturn": metrics.to_ann_return(self.value),
                "AnnVolatility": metrics.to_ann_volatility(self.value),
                "SharpeRatio": metrics.to_sharpe_ratio(self.value),
                "SortinoRatio": metrics.to_sortino_ratio(self.value),
                "CalmarRatio": metrics.to_calmar_ratio(self.value),
                "TailRatio": metrics.to_tail_ratio(self.value),
                # "JensensAlpha": metrics.to_jensens_alpha(self.value, self.prices_bm),
                # "TreynorRatio": metrics.to_treynor_ratio(self.value, self.prices_bm),
                "MaxDrawdown": metrics.to_max_drawdown(self.value),
                "Skewness": metrics.to_skewness(self.value),
                "Kurtosis": metrics.to_kurtosis(self.value),
                "VaR": metrics.to_value_at_risk(self.value),
                "CVaR": metrics.to_conditional_value_at_risk(self.value),
            }
        )

    @property
    def drawdown(self) -> pd.Series:
        return metrics.to_drawdown(self.value)
