"""ROBERT"""
from typing import Union
from typing import overload
import pandas as pd


@overload
def to_momentum(prices: pd.Series, **kwargs) -> pd.Series:
    ...


@overload
def to_momentum(prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
    ...


def to_momentum(
    prices: Union[pd.DataFrame, pd.Series], **kwargs
) -> Union[pd.DataFrame, pd.Series]:
    resampled_prices = prices.resample("D").last().ffill()
    offset_prices = resampled_prices.shift(1, freq=pd.DateOffset(**kwargs))
    return (prices / offset_prices).loc[prices.index]


