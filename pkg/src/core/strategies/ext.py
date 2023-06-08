"""ROBERT"""
import warnings
from typing import Optional, Callable, Tuple, Dict, Any, List
from functools import partial
import pandas as pd
from .base import Strategy
from ..portfolios import Optimizer


def dict_to_signature_string(data):
    # Sort dictionary by keys
    sorted_data = sorted(data.items())

    # Convert key-value pairs to a list of strings
    pairs = [f"{key}={value}" for key, value in sorted_data]

    # Join the list of strings with a delimiter
    signature_string = "&".join(pairs)

    return signature_string


class Backtest:
    def __call__(self, func) -> Callable:
        def wrapper(cls, **kwargs):
            name = kwargs.pop("name", f"Strategy-{len(cls.strategies)}")
            if name in cls.strategies:
                warnings.warn(message=f"{name} already backtested.")
                return
            strategy = Strategy(
                prices=kwargs.pop("prices", cls.prices),
                start=kwargs.pop("start", cls.start),
                end=kwargs.pop("end", cls.end),
                frequency=kwargs.pop("frequency", cls.frequency),
                commission=kwargs.pop("commission", cls.commission),
                shares_frac=kwargs.pop("shares_frac", cls.shares_frac),
                rebalance=partial(func, cls, **kwargs),
            )

            cls.strategies[name] = strategy
            return strategy

        return wrapper


class BacktestManager:
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
        try:
            from core import data

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
        except ImportError as exc:
            raise ImportError() from exc

    def reset_strategies(self) -> None:
        del self.strategies
        self.strategies = {}

    def set_universe(self, name="USSECTORETF") -> "BacktestManager":
        try:
            from ...core import data

            if name == "USSECTORETF":
                self.prices = data.get_prices(
                    tickers="XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU, GLD, BIL",
                )
            else:
                self.prices = data.get_prices("ACWI, BND")
            # self.strategies = {}
        except ImportError as exc:
            raise ImportError() from exc

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

    @Backtest()
    def Base(
        self,
        strategy: Strategy,
        objective: str = "uniform_allocation",
        feature_values: Optional[pd.DataFrame] = None,
        feature_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        optimizer_constraints: Dict[str, Tuple] = {},
        specific_constraints: List[Dict[str, Any]] = [],
    ) -> pd.Series:
        opt = Optimizer.from_prices(
            prices=strategy.prices, **optimizer_constraints
        ).set_specific_constraints(specific_constraints=specific_constraints)
        if feature_values is not None:
            if feature_bounds is None:
                warnings.warn("Must specify percentile when feature is passed.")
            current_feature_values = (
                feature_values.loc[strategy.date]
                .reindex(index=opt.assets, fill_value=0)
                .fillna(0)
            )
            opt.set_factor_constraints(
                values=current_feature_values, bounds=feature_bounds
            )
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
