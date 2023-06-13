"""ROBERT"""
from typing import Union, Iterable
from typing import overload
import pandas as pd
from ...utils import is_iterable


@overload
def to_momentum(
    prices: pd.Series, months: int = 1, skip_months: int = 0, absolute: bool = False
) -> pd.Series:
    ...


@overload
def to_momentum(
    prices: pd.DataFrame, months: int = 1, skip_months: int = 0, absolute: bool = False
) -> pd.DataFrame:
    ...


def to_momentum(
    prices: Union[pd.DataFrame, pd.Series],
    months: int = 1,
    skip_months: int = 0,
    absolute: bool = False,
) -> Union[pd.DataFrame, pd.Series]:
    cache = prices.pct_change(periods=(months - skip_months) * 21).shift(
        skip_months * 21
    )
    if absolute:
        return cache.abs()
    return cache


def to_standard_scalar(data: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    mean = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    return (data - mean) / std



def to_forward_returns(
    prices: pd.DataFrame,
    periods: Union[int, Iterable] = 21,
) -> pd.DataFrame:
    if isinstance(periods, Iterable):
        out = pd.concat(
            objs=[
                to_forward_returns(prices=prices, periods=int(period)).stack()
                for period in periods
            ],
            axis=1,
        )
        out.columns = list(periods)
        out.index.names = ["Date", "Ticker"]
        return out
    else:
        return prices.pct_change(periods=periods).shift(-periods)