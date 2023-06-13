"""ROBERT"""
import warnings
from typing import Optional, Tuple, Dict, Any, List
from functools import partial
import pandas as pd
from pkg.src.core.portfolios import Optimizer
from pkg.src import data
from .base import Strategy


class BacktestManager:

    num_strategies = 1

    @classmethod
    def from_universe(
        cls,
        name: str = "USSECTORETF",
        start: Optional[str] = None,
        end: Optional[str] = None,
        commission: int = 10,
        frequency: str = "M",
        shares_frac: Optional[int] = None,
    ) -> "BacktestManager":
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
            shares_frac=shares_frac,
        )

    def reset_strategies(self) -> None:
        del self.strategies
        self.strategies = {}

    def set_universe(self, name="USSECTORETF") -> "BacktestManager":
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
        shares_frac: Optional[int] = None,
    ) -> None:
        self.prices = prices
        self.frequency = frequency
        self.commission = commission
        self.start = start
        self.end = end
        self.shares_frac = shares_frac
        self.strategies: Dict[str, Strategy] = dict()

    def run(
        self,
        name: Optional[str] = None,
        objective: str = "uniform_allocation",
        factor_values: Optional[pd.DataFrame] = None,
        factor_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        optimizer_constraints: Optional[Dict[str, Tuple]] = None,
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Strategy:
        if name is None:
            name = f"Strategy-{self.num_strategies}"
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
            shares_frac=kwargs.pop("shares_frac", self.shares_frac),
            rebalance=partial(
                self.rebalance,
                objective=objective,
                factor_values=factor_values,
                factor_bounds=factor_bounds,
                optimizer_constraints=optimizer_constraints,
                specific_constraints=specific_constraints,
            ),
        )
        self.strategies[name] = strategy
        self.num_strategies += 1
        return strategy

    @staticmethod
    def rebalance(
        strategy: Strategy,
        objective: str = "uniform_allocation",
        factor_values: Optional[pd.DataFrame] = None,
        factor_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        optimizer_constraints: Optional[Dict[str, Tuple]] = None,
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.Series:
        """rebalance function callable"""

        if optimizer_constraints is None:
            optimizer_constraints = {}

        opt = Optimizer.from_prices(prices=strategy.prices, **optimizer_constraints)

        if specific_constraints is not None:
            opt.set_specific_constraints(specific_constraints=specific_constraints)

        if factor_values is not None:
            if factor_bounds is None:
                warnings.warn("Must specify percentile when feature is passed.")
            curr_factor_values = (
                factor_values.loc[strategy.date]
                .reindex(index=opt.assets, fill_value=0)
                .fillna(0)
            )
            opt.set_factor_constraints(values=curr_factor_values, bounds=factor_bounds)
        if not hasattr(opt, objective):
            warnings.warn(message="check you allocation objective")
            return opt.uniform_allocation()
        weight = getattr(opt, objective)()
        return weight


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
