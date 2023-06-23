"""ROBERT"""
import os
import json
from typing import Optional, Callable, Iterator, Type, Dict, List, Any, Tuple
import pandas as pd
from pkg.src.core import metrics, benchmarks
from pkg.src.core import universes, portfolios
from pkg.src.core.factors import MultiFactors
from pkg.src.core.portfolios.base import BaseOptimizer


class Rebalancer:
    def __init__(
        self,
        optimizer: str = "EqualWeight",
        factors: Optional[pd.DataFrame] = None,
        optimizer_constraints: Optional[Dict[str, float]] = None,
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
        span: Optional[int] = None,
        risk_free: float = 0.0,
        prices_bm: Optional[pd.Series] = None,
        weights_bm: Optional[pd.Series] = None,
    ) -> None:
        super().__init__()
        self.optimizer: Type[BaseOptimizer] = getattr(portfolios, optimizer)
        self.optimizer_constraints = optimizer_constraints or {}
        self.specific_constraints = specific_constraints or []
        self.factors = factors
        self.span = span
        self.risk_free = risk_free
        self.prices_bm = prices_bm
        self.weights_bm = weights_bm

    def __call__(self, strategy: "Strategy") -> pd.Series:
        """Calculate portfolio allocation weights based on the Strategy instance.

        Args:
            strategy (Strategy): Strategy instance.

        Returns:
            pd.Series: portfolio allocation weights.
        """
        opt = self.optimizer.from_prices(
            prices=strategy.prices,
            factors=self.factors.loc[strategy.date]
            .loc[strategy.prices.columns]
            .fillna(0)
            if self.factors is not None
            else None,
            span=self.span,
            risk_free=self.risk_free,
            prices_bm=self.prices_bm,
            weights_bm=self.weights_bm,
            **self.optimizer_constraints,
        ).set_specific_constraints(specific_constraints=self.specific_constraints)

        return opt.solve()


class Records(dict):
    def __init__(self, **kwargs) -> None:
        self["value"] = kwargs.get("value", {})
        self["cash"] = kwargs.get("cash", {})
        self["allocations"] = kwargs.get("allocations", {})
        self["weights"] = kwargs.get("weights", {})
        self["shares"] = kwargs.get("shares", {})
        self["trades"] = kwargs.get("trades", {})

    @property
    def performance(self) -> pd.Series:
        perf = pd.Series(self.get("value"), name="performance")
        perf.index = pd.to_datetime(perf.index)
        return perf

    @property
    def cash(self) -> pd.Series:
        return pd.Series(self.get("cash"), name="cash")

    @property
    def allocations(self) -> pd.DataFrame:
        return pd.DataFrame(self.get("allocations")).T

    @property
    def weights(self) -> pd.DataFrame:
        return pd.DataFrame(self.get("weights")).T

    @property
    def trades(self) -> pd.DataFrame:
        return pd.DataFrame(self.get("trades")).T

    ################################################################################


class Book:
    @classmethod
    def new(
        cls,
        inception: pd.Timestamp,
        initial_investment: float = 10_000.0,
    ) -> "Book":
        return cls(
            date=inception,
            value=initial_investment,
            cash=initial_investment,
            shares=pd.Series(dtype=float),
            weights=pd.Series(dtype=float),
            capitals=pd.Series(dtype=float),
        )

    def __init__(
        self,
        date: pd.Timestamp,
        value: float,
        cash: float,
        shares: pd.Series,
        weights: pd.Series,
        capitals: pd.Series,
        records: Optional[Dict] = None,
    ):
        self.date = date
        self.value = value
        self.cash = cash
        self.shares = shares if isinstance(shares, pd.Series) else pd.Series(shares)
        self.weights = weights if isinstance(weights, pd.Series) else pd.Series(weights)
        self.capitals = (
            capitals if isinstance(capitals, pd.Series) else pd.Series(capitals)
        )
        self.records = Records() if records is None else Records(**records)

    def dict(self) -> Dict:
        return {
            "date": str(self.date),
            "value": self.value,
            "cash": self.cash,
            "shares": self.shares.to_dict(),
            "weights": self.weights.to_dict(),
            "capitals": self.capitals.to_dict(),
            "records": self.records,
        }


class Strategy:
    """base strategy"""

    @classmethod
    def load(
        cls,
        name: str,
        universe: str = "UnitedStatesSectors",
        benchmark: str = "Global64",
        inception: str = "2003-1-1",
        frequency: str = "M",
        commission: int = 10,
        optimizer: str = "EqualWeight",
        min_window: int = 252,
        factors: Optional[Tuple[str]] = None,
        allow_fractional_shares: bool = False,
    ) -> "Strategy":
        # Get investment universe instance
        signiture = cls.read_file(name=name)

        if "book" in signiture:
            date = pd.Timestamp(signiture["book"]["date"])
            if (pd.Timestamp("now") - date).days <= 2:
                strategy = cls(
                    prices=pd.DataFrame(),
                    rebalance=Rebalancer(),
                    inception=signiture["inception"],
                )
                strategy.book = Book(**signiture["book"])
                return strategy

        universe_obj = getattr(
            universes, signiture.get("universe", universe)
        ).instance()

        strategy = cls(
            prices=universe_obj.prices,
            inception=signiture.get("inception", inception),
            frequency=signiture.get("frequency", frequency),
            commission=signiture.get("commission", commission),
            min_window=signiture.get("min_window", min_window),
            benchmark=signiture.get("benchmark", benchmark),
            allow_fractional_shares=signiture.get(
                "allow_fractional_shares", allow_fractional_shares
            ),
            rebalance=Rebalancer(
                optimizer=optimizer,
                factors=MultiFactors(
                    tickers=universe_obj.tickers, factors=factors
                ).standard_percentile
                if factors is not None
                else None,
            ),
        )
        if "book" in signiture:
            strategy.book = Book(**signiture["book"])
        strategy.simulate()

        return strategy

    def __init__(
        self,
        prices: pd.DataFrame,
        rebalance: Callable,
        frequency: str = "M",
        inception: Optional[str] = None,
        initial_investment: float = 10_000.0,
        commission: int = 10,
        allow_fractional_shares: bool = False,
        min_window: int = 2,
        benchmark: str = "Global64",
    ) -> None:
        self.total_prices: pd.DataFrame = prices.ffill()
        self.total_prices.index = pd.to_datetime(self.total_prices.index)
        self.inception = inception or str(self.total_prices.index[0])
        self.rebalance = rebalance
        self.commission = commission
        self.allow_fractional_shares = allow_fractional_shares
        self.initial_investment = initial_investment
        self.frequency = frequency
        self.min_window = min_window
        self.benchmark = getattr(benchmarks, benchmark).instance()

        self.book = Book(
            date=pd.Timestamp(self.inception),
            value=self.initial_investment,
            cash=self.initial_investment,
            shares=pd.Series(dtype=float),
            weights=pd.Series(dtype=float),
            capitals=pd.Series(dtype=float),
        )

    @property
    def prices(self) -> pd.DataFrame:
        """
        Get the price data for the strategy.

        Returns:
            pd.DataFrame: Price data.
        """
        return self.total_prices.loc[: self.book.date].dropna(
            thresh=self.min_window, axis=1
        )

    @property
    def date(self) -> pd.Timestamp:
        return self.book.date

    ################################################################################

    def generate_rebalance_dates(self) -> Iterator[pd.Timestamp]:
        """
        Generate rebalance dates between the given start and end dates with the specified frequency.

        Yields:
            Iterator[pd.Timestamp]: Iterator that yields rebalance dates.
        """
        for rebalance_date in pd.date_range(
            start=self.date - pd.DateOffset(days=1),
            end=pd.Timestamp("now"),
            freq=self.frequency,
        ):
            yield rebalance_date

    def generate_simulate_dates(self) -> Iterator[pd.Timestamp]:
        for simulate_date in self.total_prices.loc[self.book.date :].index:
            yield simulate_date

    def simulate(self) -> None:
        """
        Simulate the strategy.
        """

        rebalance_dates = self.generate_rebalance_dates()
        try:
            rebalance_date = next(rebalance_dates)
            while self.date > rebalance_date:
                rebalance_date = next(rebalance_dates)
        except StopIteration:
            rebalance_date = pd.Timestamp(
                str(self.total_prices.loc[self.book.date :].index[0])
            )

        for date in pd.date_range(
            start=self.date - pd.DateOffset(days=1),
            end=pd.Timestamp("now"),
            freq=self.frequency,
        ):
            self.book.capitals = self.book.shares.multiply(
                self.total_prices.loc[date]
            ).dropna()
            self.book.value = sum(self.book.capitals) + self.book.cash
            self.book.weights = self.book.capitals.divide(self.book.value)
            if date > rebalance_date or self.book.value == self.book.cash:
                try:
                    allocations = self.rebalance(strategy=self)
                    if isinstance(allocations, pd.Series):
                        self.book.records["allocations"][
                            str(self.book.date)
                        ] = allocations.to_dict()
                        target_capials = self.book.value * allocations
                        target_shares = target_capials.divide(
                            self.total_prices.loc[date]
                        )
                        target_shares = target_shares.round(
                            decimals=4 if self.allow_fractional_shares else 0
                        )
                        trade_shares = target_shares.subtract(
                            self.book.shares, fill_value=0
                        )
                        trade_shares = trade_shares[trade_shares != 0]
                        self.book.records["trades"][str(date)] = trade_shares.to_dict()
                        trade_capitals = trade_shares.multiply(
                            self.total_prices.loc[date]
                        )
                        trade_capitals += trade_capitals.multiply(
                            self.commission / 1_000
                        )
                        self.book.cash -= trade_capitals.sum()
                        self.book.shares = target_shares
                except:
                    print(f"optimization failed {self.date}")
                try:
                    if date > rebalance_date:
                        rebalance_date = next(rebalance_dates)
                except StopIteration:
                    rebalance_date = pd.Timestamp(str(self.total_prices.index[-2]))

            self.book.date = date
            self.book.records["value"][str(self.book.date)] = self.book.value
            self.book.records["cash"][str(self.book.date)] = self.book.cash
            self.book.records["shares"][
                str(self.book.date)
            ] = self.book.shares.to_dict()
            self.book.records["weights"][
                str(self.book.date)
            ] = self.book.weights.to_dict()

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

    @property
    def drawdown(self) -> pd.Series:
        return metrics.to_drawdown(self.book.records.performance)

    @classmethod
    def read_file(cls, name: str) -> Dict:
        file = os.path.join(os.path.dirname(__file__), f"{name}.json")
        try:
            with open(file=file, mode="r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    def save(self, name: str) -> None:
        signiture = {
            "benchmark": self.benchmark.__class__.__name__,
            "min_window": self.min_window,
            "inception": self.inception,
            "frequency": self.frequency,
            "commission": self.commission,
            "allow_fractional_shares": self.allow_fractional_shares,
            "book": self.book.dict(),
        }
        file = os.path.join(os.path.dirname(__file__), f"{name}.json")
        # Save the dictionary to JSON file
        with open(file=file, mode="w", encoding="utf-8") as file:
            json.dump(signiture, file)
