import os
import json
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from src.core import universes, benchmarks, metrics

"""ROBERT"""
from typing import Optional, List, Dict, Any
import pandas as pd
from src.core import portfolios
from src.core.factors import MultiFactor

logger = logging.getLogger(__name__)


class Rebalancer:
    def __call__(self, strategy: "Strategy") -> pd.Series:
        """Calculate portfolio allocation weights based on the Strategy instance.

        Args:
            strategy (Strategy): Strategy instance.

        Returns:
            pd.Series: portfolio allocation weights.
        """
        prices = strategy.reb_prices
        factors = strategy.factor.get_factor_by_date(
            tickers=list(prices.columns), date=strategy.date
        )
        opt = strategy.portfolio.from_prices(
            prices=prices,
            span=None,
            factors=factors,
            # weights_bm=pd.Series(0, index=prices.columns),
            # prices_bm=strategy.benchmark.get_performance(strategy.date),
            # weights_bm=strategy.benchmark.get_weights(strategy.date),
            # sum_weight=0.0, min_weight = -1, max_weight = 1,
            weights_bm=pd.Series(1 / len(prices.columns), index=prices.columns),
            **strategy.portfolio_constraints,
        ).set_specific_constraints(strategy.specific_constraints)
        return opt.solve()


class Records(dict):
    def __init__(self, **kwargs) -> None:
        for attr in [
            "value",
            "cash",
            "weights",
            "shares",
            "capitals",
            "allocations",
            "trades",
        ]:
            self[attr] = kwargs.get(attr, {})

    @property
    def performance(self) -> pd.Series:
        perf = pd.Series(self["value"], name="performance")
        perf.index = pd.to_datetime(perf.index)
        return perf

    @property
    def cash(self) -> pd.Series:
        return pd.Series(self["cash"], name="cash")

    @property
    def allocations(self) -> pd.DataFrame:
        return pd.DataFrame(self["allocations"]).T

    @property
    def weights(self) -> pd.DataFrame:
        return pd.DataFrame(self["weights"]).T

    @property
    def trades(self) -> pd.DataFrame:
        return pd.DataFrame(self["trades"]).T


class Book:
    def __init__(
        self,
        date: pd.Timestamp,
        value: float = 0.,
        cash: float = 0.,
        shares=pd.Series(dtype=float),
        capitals=pd.Series(dtype=float),
        allocations=pd.Series(dtype=float),
        records: Optional[Dict] = None,
    ):
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
        self.date = date
        self.value = value
        self.cash = cash
        if not isinstance(shares, pd.Series):
            shares = pd.Series(shares)
        self.shares = shares
        if not isinstance(capitals, pd.Series):
            capitals = pd.Series(capitals)
        self.capitals = capitals
        if not isinstance(allocations, pd.Series):
            allocations = pd.Series(allocations)
        self.allocations = allocations
        self.records = Records(**(records or {}))

    def dict(self) -> Dict:
        return {
            "date": str(self.date),
            "value": self.value,
            "cash": self.cash,
            "shares": self.shares.to_dict(),
            "capitals": self.capitals.to_dict(),
            "allocations": self.allocations.to_dict(),
            "records": self.records,
        }

    def is_empty(self) -> bool:
        return self.cash == self.value


class Strategy:
    def __init__(
        self,
        rebalancer: Rebalancer,
        universe: universes.Universe,
        inception: Optional[str] = None,
        frequency: str = "M",
        initial_investment: int = 10_000,
        commission: int = 10,
        allow_fractional_shares: bool = False,
        min_window: int = 1,
        portoflio: portfolios.Portfolio = portfolios.EqualWeight(),
        factor: MultiFactor = MultiFactor(),
        portfolio_constraints: Optional[Dict[str, float]] = None,
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.rebalancer = rebalancer
        self.universe = universe
        self.inception = inception or self.universe.inception
        self.frequency = frequency
        self.min_window = min_window
        self.commission = commission
        self.initial_investment = initial_investment
        self.allow_fractional_shares = allow_fractional_shares
        self.portfolio = portoflio
        self.factor = factor
        self.portfolio_constraints = portfolio_constraints or {}
        self.specific_constraints = specific_constraints or []
        self.book = Book(
            date=pd.Timestamp(self.inception),
        )
        logger.warning("Strategy init finished.")
        self.prices = pd.DataFrame()

    @property
    def reb_prices(self) -> pd.DataFrame:
        """
        Get the price data for the strategy.

        Returns:
            pd.DataFrame: Price data.
        """
        return self.prices.loc[: self.book.date].dropna(thresh=self.min_window, axis=1)

    @property
    def date(self) -> pd.Timestamp:
        return self.book.date

    def simulate(self) -> "Strategy":
        end = pd.Timestamp("now") + pd.DateOffset(days=1)
        start = self.date + pd.DateOffset(days=1)
        reb_dates = pd.date_range(start=start, end=end, freq=self.frequency)
        if len(reb_dates) == 0:
            return self
        self.prices = self.universe.get_prices()
        for self.book.date in pd.date_range(start=start, end=end):
            if self.book.date in self.prices.index:
                if self.book.value == 0.0:
                    self.book.value = self.book.cash = self.initial_investment
                else:
                    # update book here
                    prices_now = self.prices.loc[self.book.date]
                    self.book.capitals = self.book.shares.multiply(prices_now).dropna()
                    self.book.value = sum(self.book.capitals) + self.book.cash
                    # make trade here
                    if not self.book.allocations.empty:
                        # Calculate trade shares.
                        target_capitals = self.book.value * self.book.allocations
                        target_shares = target_capitals.divide(prices_now)
                        decimals = decimals = 4 if self.allow_fractional_shares else 0
                        target_shares = target_shares.round(decimals)
                        trade_shares = target_shares.subtract(
                            self.book.shares, fill_value=0
                        )
                        trade_shares = trade_shares[trade_shares != 0]
                        # Store allocations & trades.
                        self.book.records["trades"][str(self.date)] = trade_shares.to_dict()
                        trade_capitals = trade_shares.multiply(prices_now)
                        trade_cost = trade_capitals.abs().sum() * (self.commission / 10_000)
                        self.book.cash -= trade_capitals.sum()
                        self.book.cash -= trade_cost
                        self.book.shares = target_shares
                        self.book.capitals = self.book.shares.multiply(prices_now).dropna()
                        self.book.value = sum(self.book.capitals) + self.book.cash
                        self.book.allocations = pd.Series(dtype=float)
                self.book.records["value"][str(self.date)] = self.book.value
                self.book.records["cash"][str(self.date)] = self.book.cash
                self.book.records["shares"][str(self.date)] = self.book.shares.to_dict()
                self.book.records["capitals"][
                    str(self.date)
                ] = self.book.capitals.to_dict()

            if self.book.allocations.empty:
                if self.book.is_empty() or self.date in reb_dates:
                    self.book.allocations = self.rebalancer(strategy=self)
                    self.book.records["allocations"][
                        str(self.date)
                    ] = self.book.allocations.to_dict()
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

    def save(self, name: str, override: bool = False) -> bool:
        signature = self.get_signature()
        file_path = Path(os.path.dirname(__file__)) / "db" / f"{name}.json"

        # Here add a block to check if the files already exist, if it exits do not save.
        if not override:
            if file_path.exists():
                print(f"File {file_path} already exists. Not saving.")
                return False
        try:
            with open(file=file_path, mode="w", encoding="utf-8") as file:
                json.dump(signature, file)
            return True
        except OSError as e:
            print(f"Error occurred while saving the file: {e}")
            if file_path.exists():
                file_path.unlink()
        return False

    def get_signature(self) -> Dict:
        return {
            "universe": self.universe.__class__.__name__,
            "portfolio": self.portfolio.__class__.__name__,
            "min_window": self.min_window,
            "inception": self.inception,
            "frequency": self.frequency,
            "commission": self.commission,
            "allow_fractional_shares": self.allow_fractional_shares,
            "portfolio_constraints": self.portfolio_constraints,
            "specific_constraints": self.specific_constraints,
            "factor": tuple(self.factor.keys()),
            "book": self.book.dict(),
        }

    @property
    def performance(self) -> pd.Series:
        out = pd.Series(self.book.records["value"], name="Performance")
        out.index = pd.to_datetime(out.index)
        return out

    @property
    def cash(self) -> pd.Series:
        out = pd.Series(self.book.records["cash"], name="cash")
        out.index = pd.to_datetime(out.index)
        return out

    @property
    def allocations(self) -> pd.DataFrame:
        out = pd.DataFrame(self.book.records["allocations"]).T
        out.index = pd.to_datetime(out.index)
        return out

    @property
    def weights(self) -> pd.DataFrame:
        out = pd.DataFrame(self.book.records["weights"]).T
        out.index = pd.to_datetime(out.index)
        return out

    @property
    def trades(self) -> pd.DataFrame:
        out = pd.DataFrame(self.book.records["trades"]).T
        out.index = pd.to_datetime(out.index)
        return out

    @property
    def drawdown(self) -> pd.Series:
        out = metrics.to_drawdown(self.performance)
        out.name = "drawdonw"
        return out
