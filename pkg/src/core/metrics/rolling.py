"""ROBERT"""
from typing import Union
from typing import overload
import pandas as pd


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


