"""ROBERT"""
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from pkg.src.core import portfolios
from .strategy import Strategy, Rebalancer


class MultiStrategy(dict):
    def run(
        self,
        prices: pd.DataFrame,
        optimizer: str = "EqualWeight",
        benchmark: str = "Global64,",
        inception: Optional[str] = None,
        factors: Optional[pd.DataFrame] = None,
        optimizer_constraints: Optional[Dict[str, float]] = None,
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
        frequency: str = "M",
        commission: int = 10,
        min_window: int = 2,
        allow_fractional_shares: bool = False,
        name: Optional[str] = None,
    ) -> Strategy:
        if name is None:
            name = f"Strategy-{len(self) + 1}"
            if name in self:
                raise ValueError("strategy `{name}` already backtested.")
        if optimizer_constraints is None:
            optimizer_constraints = {}
        if specific_constraints is None:
            specific_constraints = []

        strategy = Strategy(
            prices=prices,
            inception=inception,
            frequency=frequency,
            commission=commission,
            min_window=min_window,
            allow_fractional_shares=allow_fractional_shares,
            rebalance=Rebalancer(
                optimizer=optimizer,
                factors=factors,
                optimizer_constraints=optimizer_constraints,
                specific_constraints=specific_constraints,
            ),
        )
        strategy.simulate()
        self[name] = strategy
        return strategy

    @property
    def performance(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: strategy.book.records.performance for name, strategy in self.items()}
        )

    @property
    def drawdowns(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: strategy.book.records.drawdown for name, strategy in self.items()}
        )

    @property
    def analytics(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: strategy.analytics for name, strategy in self.items()}
        )

    def load(
        self,
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
        optimizer_constraints: Optional[Dict[str, float]] = None,
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
    ):
        if optimizer_constraints is None:
            optimizer_constraints = {}
        if specific_constraints is None:
            specific_constraints = []
        strategy = Strategy.load(
            name=name,
            universe=universe,
            benchmark=benchmark,
            inception=inception,
            frequency=frequency,
            commission=commission,
            optimizer=optimizer,
            min_window=min_window,
            factors=factors,
            allow_fractional_shares=allow_fractional_shares,
        )
        self[name] = strategy
        return strategy
