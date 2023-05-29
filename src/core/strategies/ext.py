"""ROBERT"""
import warnings
from typing import Optional, Callable, Union, Dict, List
from functools import partial
import pandas as pd
from .base import Strategy
from .. import metrics
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
            name = (
                func.__name__ + "(" + dict_to_signature_string(kwargs) + ")"
                if kwargs
                else func.__name__
            )
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
    def RegimeRotation(self, strategy: Strategy, signal) -> pd.Series:
        state = signal.get_state(str(strategy.date))
        exp_ret = signal.expected_returns_by_states(strategy.prices.iloc[-10 * 252 :])
        index = exp_ret.loc[state].nlargest(5).index
        return Optimizer.from_prices(prices=strategy.prices[index]).uniform_allocation()

    @Backtest()
    def RegimeRotationMinCorr(self, strategy: Strategy, signal, **kwargs) -> pd.Series:
        state = signal.get_state(str(strategy.date))
        exp_ret = signal.expected_returns_by_states(strategy.prices.iloc[-10 * 252 :])
        index = exp_ret.loc[state].nlargest(5).index
        return Optimizer.from_prices(
            prices=strategy.prices[index], **kwargs
        ).minimized_correlation()

    @Backtest()
    def EqualWeight(self, strategy: Strategy, **kwargs) -> pd.Series:
        """equal"""
        return Optimizer.from_prices(
            prices=strategy.prices, **kwargs
        ).uniform_allocation()

    @Backtest()
    def Momentum(
        self,
        strategy: Strategy,
        target_score: float = 0.8,
        months: Union[int, List[int]] = 12,
        objective: str = "uniform_allocation",
    ) -> pd.Series:
        if isinstance(months, list):
            rank = (
                pd.concat(
                    [
                        metrics.to_momentum(prices=strategy.prices, months=m)
                        for m in months
                    ],
                    axis=1,
                )
                .rank(ascending=True, pct=True)
                .mean(axis=1)
                .rank(pct=True, ascending=True)
            )
        else:
            mom = metrics.to_momentum(prices=strategy.prices, months=months)
            rank = mom.rank(pct=True, ascending=True)
        opt = Optimizer.from_prices(
            prices=strategy.prices
        ).set_custom_feature_constraints(
            features=rank, min_value=target_score, max_value=target_score
        )
        return getattr(opt, objective)()

    @Backtest()
    def MinCorr(self, strategy: Strategy, **kwargs) -> pd.Series:
        return Optimizer.from_prices(
            prices=strategy.prices, **kwargs
        ).minimized_correlation()

    @Backtest()
    def MinVol(self, strategy: Strategy, **kwargs) -> pd.Series:
        return Optimizer.from_prices(
            prices=strategy.prices, **kwargs
        ).minimized_volatility()

    @Backtest()
    def MaxSharpe(self, strategy: Strategy, **kwargs) -> pd.Series:
        return Optimizer.from_prices(
            prices=strategy.prices, **kwargs
        ).maximized_sharpe_ratio()

    @Backtest()
    def RiskParity(self, strategy: Strategy, **kwargs) -> pd.Series:
        return Optimizer.from_prices(prices=strategy.prices, **kwargs).risk_parity()

    @Backtest()
    def HRiskParity(self, strategy: Strategy, **kwargs) -> pd.Series:
        return Optimizer.from_prices(
            prices=strategy.prices, **kwargs
        ).hierarchical_risk_parity()

    @Backtest()
    def MeanReversion(self, strategy: Strategy, threshold: float = 0.20) -> pd.Series:
        sma = strategy.prices.rolling(50).mean()
        deviation = strategy.prices / sma
        rr = deviation.iloc[-1] - deviation.quantile(q=threshold)
        rr = rr[rr < 0]
        if len(rr) == 0:
            return pd.Series(dtype=float)
        rr.iloc[:] = 1 / rr.count()
        return rr

    @property
    def values(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: strategy.value for name, strategy in self.strategies.items()}
        )

    @property
    def analytics(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: strategy.analytics for name, strategy in self.strategies.items()}
        )
