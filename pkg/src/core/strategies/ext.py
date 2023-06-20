"""ROBERT"""
import warnings
from typing import Optional, Tuple, Dict, Any, List, Type
import pandas as pd
from pkg.src.core import portfolios
from pkg.src import data
from .base import Strategy

class Rebalance:
    def __init__(
        self,
        optimizer: Type[portfolios.base.BaseOptimizer] = portfolios.EqualWeight,
        factor_values: Optional[pd.DataFrame] = None,
        factor_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        optimizer_constraints: Optional[Dict[str, float]] = None,
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
        span: Optional[int] = None,
        risk_free: float = 0.0,
        prices_bm: Optional[pd.Series] = None,
        weights_bm: Optional[pd.Series] = None,
    ) -> None:
        self.optimizer = optimizer
        self.optimizer_constraints = optimizer_constraints or {}
        self.specific_constraints = specific_constraints or []
        self.factor_values = factor_values
        self.factor_bounds = factor_bounds
        self.span = span
        self.risk_free = risk_free
        self.prices_bm = prices_bm
        self.weights_bm = weights_bm

    def __call__(self, strategy: Strategy) -> pd.Series:
        """Calculate portfolio allocation weights based on the Strategy instance.

        Args:
            strategy (Strategy): Strategy instance.

        Returns:
            pd.Series: portfolio allocation weights.
        """
        opt = self.optimizer.from_prices(
            prices=strategy.prices,
            span=self.span,
            risk_free=self.risk_free,
            prices_bm=self.prices_bm,
            weights_bm=self.weights_bm,
            **self.optimizer_constraints,
        ).set_specific_constraints(specific_constraints=self.specific_constraints)

        if self.factor_values is not None:
            if self.factor_bounds is None:
                warnings.warn("Must specify percentile when feature is passed.")
            curr_factor_values = (
                self.factor_values.loc[strategy.date]
                .reindex(index=opt.assets, fill_value=0)
                .fillna(0)
            )
            opt.set_factor_constraints(
                values=curr_factor_values, bounds=self.factor_bounds
            )
        return opt.solve()


class MultiStrategy:
    num_strategies = 0

    @classmethod
    def from_universe(
        cls,
        name: str = "USSECTORETF",
        start: Optional[str] = None,
        end: Optional[str] = None,
        commission: int = 10,
        frequency: str = "M",
        allow_fractional_shares: bool = False,
    ) -> "MultiStrategy":
        if name == "USSECTORETF":
            prices = data.get_prices(
                tickers="XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU, GLD, BIL",
            )
        else:
            prices = data.get_prices("ACWI, BND")
        return cls(
            prices=prices,
            start=start,
            end=end,
            commission=commission,
            frequency=frequency,
            allow_fractional_shares=allow_fractional_shares,
        )

    def reset_strategies(self) -> None:
        del self.strategies
        self.strategies = {}

    def set_universe(self, name="USSECTORETF") -> "MultiStrategy":
        if name == "USSECTORETF":
            self.prices = data.get_prices(
                tickers="XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU, GLD, BIL",
            )
        else:
            self.prices = data.get_prices("ACWI, BND")
        return self

    def __init__(
        self,
        prices: Optional[pd.DataFrame] = None,
        frequency: str = "M",
        commission: int = 10,
        start: Optional[str] = None,
        end: Optional[str] = None,
        allow_fractional_shares: bool = False,
    ) -> None:
        self.prices = prices
        self.frequency = frequency
        self.commission = commission
        self.start = start
        self.end = end
        self.allow_fractional_shares = allow_fractional_shares
        self.strategies: Dict[str, Strategy] = dict()

    def run(
        self,
        name: Optional[str] = None,
        optimizer: str = portfolios.EqualWeight.__name__,
        factor_values: Optional[pd.DataFrame] = None,
        factor_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        optimizer_constraints: Optional[Dict[str, float]] = None,
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Strategy:
        if name is None:
            name = f"Strategy-{self.num_strategies + 1}"
            if name in self.strategies:
                raise ValueError("strategy `{name}` already backtested.")
        if optimizer_constraints is None:
            optimizer_constraints = {}
        if specific_constraints is None:
            specific_constraints = []

        strategy = Strategy(
            prices=kwargs.pop("prices", self.prices),
            start=kwargs.pop("start", self.start),
            end=kwargs.pop("end", self.end),
            frequency=kwargs.pop("frequency", self.frequency),
            commission=kwargs.pop("commission", self.commission),
            allow_fractional_shares=kwargs.pop(
                "allow_fractional_shares", self.allow_fractional_shares
            ),
            rebalance=Rebalance(
                optimizer=getattr(portfolios, optimizer),
                factor_values=factor_values,
                factor_bounds=factor_bounds,
                optimizer_constraints=optimizer_constraints,
                specific_constraints=specific_constraints,
            ),
        )
        self.strategies[name] = strategy
        self.num_strategies += 1
        return strategy

    @property
    def values(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: strategy.value for name, strategy in self.strategies.items()}
        )

    @property
    def drawdowns(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: strategy.drawdown for name, strategy in self.strategies.items()}
        )

    @property
    def analytics(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: strategy.analytics for name, strategy in self.strategies.items()}
        )

    def drop_strategy(self, name: str) -> None:
        if name not in self.strategies:
            warnings.warn(message=f"no strategy named {name}")
            return
        del self.strategies[name]
