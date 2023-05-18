"""ROBERT"""
from typing import Optional, Callable, Dict
from functools import partial
import pandas as pd
from .base import Strategy
from ..portfolios import Optimizer


class Backtest:
    def __call__(self, func) -> Callable:
        def wrapper(
            cls,
            prices: Optional[pd.DataFrame] = None,
            start: Optional[str] = None,
            end: Optional[str] = None,
            **kwargs,
        ):
            strategy = Strategy(
                prices=cls.prices if prices is None else prices,
                rebalance=partial(func, cls, **kwargs),
                start=start or cls.start,
                end=end or cls.end,
            )

            cls.strategies[func.__name__] = strategy

            return strategy

        return wrapper


class BacktestManager:
    strategies: Dict[str, Strategy] = dict()

    def __init__(
        self,
        prices: Optional[pd.DataFrame] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> None:
        self.prices = prices
        self.start = start
        self.end = end

    @Backtest()
    def RegimeRotation(self, strategy: Strategy, signal) -> pd.Series:
        state = signal.get_state(str(strategy.date))
        exp_ret = signal.expected_returns_by_states(
            strategy.reb_prices.iloc[-10 * 252 :]
        )
        index = exp_ret.loc[state].nlargest(5).index
        return Optimizer.from_prices(
            prices=strategy.reb_prices[index]
        ).uniform_allocation()

    @Backtest()
    def RegimeRotationMinCorr(self, strategy: Strategy, signal) -> pd.Series:
        state = signal.get_state(str(strategy.date))
        exp_ret = signal.expected_returns_by_states(
            strategy.reb_prices.iloc[-10 * 252 :]
        )
        index = exp_ret.loc[state].nlargest(5).index
        return Optimizer.from_prices(
            prices=strategy.reb_prices[index]
        ).minimized_correlation()

    @Backtest()
    def EqualWeight(self, strategy: Strategy) -> pd.Series:
        """equal"""
        return Optimizer.from_prices(prices=strategy.reb_prices).uniform_allocation()

    @Backtest()
    def Momentum(self, strategy: Strategy) -> pd.Series:
        mom = strategy.reb_prices.iloc[-1] / strategy.reb_prices.iloc[-252]
        index = mom.dropna().nlargest(5).index
        return Optimizer.from_prices(
            prices=strategy.reb_prices[index]
        ).uniform_allocation()

    @Backtest()
    def MinCorr(self, strategy: Strategy) -> pd.Series:
        return Optimizer.from_prices(prices=strategy.reb_prices).minimized_correlation()

    @Backtest()
    def MinVol(self, strategy: Strategy) -> pd.Series:
        return Optimizer.from_prices(prices=strategy.reb_prices).minimized_volatility()

    @Backtest()
    def RiskParity(self, strategy: Strategy) -> pd.Series:
        return Optimizer.from_prices(prices=strategy.reb_prices).risk_parity()

    @Backtest()
    def HRiskParity(self, strategy: Strategy) -> pd.Series:
        return Optimizer.from_prices(
            prices=strategy.reb_prices
        ).hierarchical_risk_parity()

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
