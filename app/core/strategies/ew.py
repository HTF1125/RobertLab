from typing import Optional
import pandas as pd
from .base import Strategy
from ..portfolios import Optimizer


def equal_weights(prices: pd.DataFrame, start: Optional[str]=None) -> Strategy:
    """equal weights"""

    def rebalance(prices: pd.DataFrame, **kwargs) -> Optional[pd.Series]:
        return Optimizer.from_prices(prices=prices).uniform_allocation()

    return Strategy(prices=prices, rebalance=rebalance).simulate(start=start)
