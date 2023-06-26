"""ROBERT"""
import os
import json
from typing import Optional, Callable, Dict, Union
import pandas as pd
from src.backend.core import benchmarks, universes, metrics
from .book import Book

INITIAL_INVESTMENT = 10_000
MIN_WINDOW = 1
FREQUENCY = "M"
COMMISSION = 10
ALLOW_FRACTIONAL_SHARES = False


class ModuleParser:
    @staticmethod
    def parse_universe(
        universe: Optional[Union[str, universes.Universe]]
    ) -> universes.Universe:
        """Parse the universe to ensure it is an instance of the Universe class.

        Args:
            universe (Optional[Union[str, universes.Universe]]): The universe to parse.

        Returns:
            universes.Universe: The parsed universe instance.

        Raises:
            ValueError: If the universe is not an instance of the Universe class.

        """
        if isinstance(universe, str):
            universe = getattr(universes, universe)()
        if not isinstance(universe, universes.Universe):
            raise ValueError(
                "Parsing universe failed. Available universes: ", universes.__all__
            )
        return universe

    @staticmethod
    def parse_benchmark(
        benchmark: Optional[Union[str, benchmarks.Benchmark]],
    ) -> benchmarks.Benchmark:
        """Parse the benchmark to ensure it is an instance of the Benchmark class.

        Args:
            benchmark (Optional[Union[str, benchmarks.Benchmark]]): The benchmark to parse.

        Returns:
            benchmarks.Benchmark: The parsed benchmark instance.

        Raises:
            ValueError: If the benchmark is not an instance of the Benchmark class.

        """
        if isinstance(benchmark, str):
            benchmark = getattr(benchmarks, benchmark)()
        if not isinstance(benchmark, benchmarks.Benchmark):
            raise ValueError(
                "Parsing benchmark failed. Available benchmarks: ", benchmarks.__all__
            )
        return benchmark


class Strategy(ModuleParser):
    """
    Strategy is an algorithmic trading strategy that sequentially allocates
    capital among a group of assets based on a pre-defined rebalance method.

    Strategy shall be the parent for all investment strategies with a period-wise
    rebalancing scheme.
    """

    @classmethod
    def from_universe(
        cls,
        universe: Optional[Union[str, universes.Universe]],
        rebalance: Callable,
        frequency: str = FREQUENCY,
        initial_investment: float = INITIAL_INVESTMENT,
        commission: int = COMMISSION,
        allow_fractional_shares: bool = ALLOW_FRACTIONAL_SHARES,
        min_window: int = MIN_WINDOW,
        inception: Optional[str] = None,
        benchmark: Optional[Union[str, benchmarks.Benchmark]] = None,
    ) -> "Strategy":
        """Create a Strategy instance from a predefined universe.

        Args:
            universe (Optional[Union[str, universes.Universe]]): The universe to use for the strategy.
            rebalance (Callable): The rebalance method to apply.
            frequency (str, optional): The frequency of rebalancing. Defaults to FREQUENCY.
            initial_investment (float, optional): The initial investment amount. Defaults to INITIAL_INVESTMENT.
            commission (int, optional): The commission amount. Defaults to COMMISSION.
            allow_fractional_shares (bool, optional): Whether to allow fractional shares. Defaults to ALLOW_FRACTIONAL_SHARES.
            min_window (int, optional): The minimum window size. Defaults to MIN_WINDOW.
            inception (Optional[str], optional): The inception date of the strategy. Defaults to None.
            benchmark (Optional[Union[str, benchmarks.Benchmark]], optional): The benchmark to compare the strategy against. Defaults to None.

        Returns:
            Strategy: An instance of the Strategy class.

        """
        universe = cls.parse_universe(universe=universe)
        prices = universe.get_prices()
        return Strategy(
            prices=prices,
            rebalance=rebalance,
            frequency=frequency,
            initial_investment=initial_investment,
            commission=commission,
            allow_fractional_shares=allow_fractional_shares,
            min_window=min_window,
            inception=inception,
            universe=universe,
            benchmark=benchmark,
        )

    def __init__(
        self,
        prices: pd.DataFrame,
        rebalance: Callable,
        frequency: str = FREQUENCY,
        initial_investment: float = INITIAL_INVESTMENT,
        commission: int = COMMISSION,
        allow_fractional_shares: bool = ALLOW_FRACTIONAL_SHARES,
        min_window: int = MIN_WINDOW,
        inception: Optional[str] = None,
        universe: Optional[Union[str, universes.Universe]] = None,
        benchmark: Optional[Union[str, benchmarks.Benchmark]] = None,
    ) -> None:
        """Initialize the Strategy class.

        Args:
            prices (pd.DataFrame): The prices of assets.
            rebalance (Callable): The rebalance method to apply.
            frequency (str, optional): The frequency of rebalancing. Defaults to FREQUENCY.
            initial_investment (float, optional): The initial investment amount. Defaults to INITIAL_INVESTMENT.
            commission (int, optional): The commission amount. Defaults to COMMISSION.
            allow_fractional_shares (bool, optional): Whether to allow fractional shares. Defaults to ALLOW_FRACTIONAL_SHARES.
            min_window (int, optional): The minimum window size. Defaults to MIN_WINDOW.
            inception (Optional[str], optional): The inception date of the strategy. Defaults to None.
            universe (Optional[Union[str, universes.Universe]], optional): The universe to use for the strategy. Defaults to None.
            benchmark (Optional[Union[str, benchmarks.Benchmark]], optional): The benchmark to compare the strategy against. Defaults to None.

        """
        self.prices: pd.DataFrame = prices.ffill()
        self.prices.index = pd.to_datetime(self.prices.index)
        self.inception = inception or str(self.prices.index[0])
        self.rebalance = rebalance
        self.commission = commission
        self.allow_fractional_shares = allow_fractional_shares
        self.initial_investment = initial_investment
        self.frequency = frequency
        self.min_window = min_window
        self.universe = None if universe is None else self.parse_universe(universe)
        self.benchmark = None if benchmark is None else self.parse_benchmark(benchmark)
        self.book = Book(
            date=pd.Timestamp(self.inception),
            value=self.initial_investment,
            cash=self.initial_investment,
            shares=pd.Series(dtype=float),
            weights=pd.Series(dtype=float),
            capitals=pd.Series(dtype=float),
        )

    @property
    def reb_prices(self) -> pd.DataFrame:
        """
        Get the price data for the strategy.

        Returns:
            pd.DataFrame: Price data.
        """
        return self.prices.loc[: self.book.date].dropna(
            thresh=self.min_window, axis=1
        )

    @property
    def date(self) -> pd.Timestamp:
        """
        Get the current date of the strategy.

        Returns:
            pd.Timestamp: Current date.
        """
        return self.book.date

    ################################################################################

    def update_book(self, date: pd.Timestamp) -> None:
        """
        Update the capitals based on current shares and prices.

        Args:
            date (pd.Timestamp): Current date.
        """
        prices_now = self.prices.loc[date]
        self.book.capitals = self.book.shares.multiply(prices_now).dropna()
        self.book.value = sum(self.book.capitals) + self.book.cash

    def needs_rebalance(self, date: pd.Timestamp, rebalance_date: pd.Timestamp) -> bool:
        """
        Check if the strategy needs rebalancing.

        Args:
            date (pd.Timestamp): Current date.
            rebalance_date (pd.Timestamp): Date of the last rebalance.

        Returns:
            bool: True if rebalancing is needed, False otherwise.
        """
        return date > rebalance_date or self.book.value == self.book.cash

    def call_reblance(self, date: pd.Timestamp) -> None:
        """
        Rebalance the strategy by executing trades based on the rebalance function.

        Args:
            date (pd.Timestamp): Current date.
        """
        allocations = self.rebalance(strategy=self)
        if isinstance(allocations, pd.Series):
            prices = self.prices.loc[date]
            # Calculate trade shares.
            target_capitals = self.book.value * allocations
            target_shares = target_capitals.divide(prices)
            decimals = decimals = 4 if self.allow_fractional_shares else 0
            target_shares = target_shares.round(decimals)
            trade_shares = target_shares.subtract(self.book.shares, fill_value=0)
            trade_shares = trade_shares[trade_shares != 0]

            # Store allocations & trades.
            # self.book.records
            self.book.records["allocations"][str(self.date)] = allocations.to_dict()
            self.book.records["trades"][str(date)] = trade_shares.to_dict()

            trade_capitals = trade_shares.multiply(prices)
            trade_capitals += trade_capitals.multiply(self.commission / 1_000)
            self.book.cash -= trade_capitals.sum()
            self.book.shares = target_shares

    def simulate(self) -> "Strategy":
        """
        Simulate the strategy.
        """
        rebalance_date = self.date - pd.DateOffset(days=1)
        for date in self.prices.loc[self.date :].index:
            self.update_book(date)
            if self.needs_rebalance(date, rebalance_date):
                try:
                    self.call_reblance(date=date)
                    rebalance_date = pd.date_range(
                        start=date, freq=self.frequency, periods=1
                    )[0]
                except Exception as e:
                    print(f"Optimization failed at {date}: {str(e)}")

            self.book.date = date
        return self

    @property
    def analytics(self) -> pd.Series:
        """
        Get the analytics of the strategy.

        Returns:
            pd.Series: Strategy analytics.
        """
        return pd.Series(
            data={
                "AnnReturn": metrics.to_ann_return(self.book.records.performance),
                "AnnVolatility": metrics.to_ann_volatility(
                    self.book.records.performance
                ),
                "SharpeRatio": metrics.to_sharpe_ratio(self.book.records.performance),
                "SortinoRatio": metrics.to_sortino_ratio(self.book.records.performance),
                "CalmarRatio": metrics.to_calmar_ratio(self.book.records.performance),
                "TailRatio": metrics.to_tail_ratio(self.book.records.performance),
                # "JensensAlpha": metrics.to_jensens_alpha(self.value, self.prices_bm),
                # "TreynorRatio": metrics.to_treynor_ratio(self.value, self.prices_bm),
                "MaxDrawdown": metrics.to_max_drawdown(self.book.records.performance),
                "Skewness": metrics.to_skewness(self.book.records.performance),
                "Kurtosis": metrics.to_kurtosis(self.book.records.performance),
                "VaR": metrics.to_value_at_risk(self.book.records.performance),
                "CVaR": metrics.to_conditional_value_at_risk(
                    self.book.records.performance
                ),
            }
        )


    @classmethod
    def read_file(cls, name: str) -> Dict:
        file = os.path.join(os.path.dirname(__file__), "db", f"{name}.json")
        try:
            with open(file=file, mode="r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    def save(self, name: str) -> None:
        signiture = {
            "universe" : str(self.universe),
            "benchmark": self.benchmark.__class__.__name__,
            "min_window": self.min_window,
            "inception": self.inception,
            "frequency": self.frequency,
            "commission": self.commission,
            "allow_fractional_shares": self.allow_fractional_shares,
            "book": self.book.dict(),
        }
        file = os.path.join(os.path.dirname(__file__), "db", f"{name}.json")
        # Save the dictionary to JSON file
        with open(file=file, mode="w", encoding="utf-8") as file:
            json.dump(signiture, file)


    @property
    def performance(self) -> pd.Series:
        out = pd.Series(self.book.records["value"], name="performance")
        out.index = pd.to_datetime(out.index)
        return out

    @property
    def cash(self) -> pd.Series:
        out = pd.Series(self.book.records["cash"], name="cash")
        out.index = pd.to_datetime(out.index)
        return out

    @property
    def allocations(self) -> pd.DataFrame:
        out= pd.DataFrame(self.book.records["allocations"]).T
        out.index = pd.to_datetime(out.index)
        return out

    @property
    def weights(self) -> pd.DataFrame:
        out= pd.DataFrame(self.book.records["weights"]).T
        out.index = pd.to_datetime(out.index)
        return out

    @property
    def trades(self) -> pd.DataFrame:
        out= pd.DataFrame(self.book.records["trades"]).T
        out.index = pd.to_datetime(out.index)
        return out

    @property
    def drawdown(self) -> pd.Series:
        out = metrics.to_drawdown(self.performance)
        out.name = "drawdonw"
        return out
