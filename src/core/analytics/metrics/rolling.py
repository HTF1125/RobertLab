import numpy as np
import pandas as pd
from . import base
from ...ext.periods import AnnFactor


def rolling(func):
    """apply vectorized prices"""

    def wrapper(prices, window: int = 252, *args, **kwargs):
        return prices.rolling(window).apply(
            getattr(base, func.__name__), args=args, kwargs=kwargs
        )

    return wrapper


def to_ann_return(
    prices: pd.DataFrame, window: int = 252, ann_factor: float = AnnFactor.daily
) -> pd.DataFrame:
    return np.exp(base.to_log_return(prices).rolling(window).mean() * ann_factor) - 1


@rolling
def to_ann_variance() -> pd.DataFrame:
    ...


@rolling
def to_ann_volatility() -> pd.DataFrame:
    ...


@rolling
def to_sharpe_ratio() -> pd.DataFrame:
    ...
