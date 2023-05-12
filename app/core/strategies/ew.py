"""ROBERT"""
from typing import Optional
import pandas as pd
from .base import Strategy
from ..portfolios import Optimizer


def equal_weights(prices: pd.DataFrame, start: Optional[str] = None) -> Strategy:
    """equal weights"""

    def rebalance(strategy: Strategy) -> Optional[pd.Series]:
        """Default rebalancing method"""
        reb_prices = strategy.reb_prices
        if reb_prices is None:
            return None
        return Optimizer.from_prices(prices=reb_prices).uniform_allocation()

    return Strategy(prices=prices, rebalance=rebalance).simulate(start=start)
