

from typing import Union
from functools import singledispatch
import pandas as pd

@singledispatch
def to_start(prices: Union[pd.DataFrame, pd.Series]):
    """get the start of the data"""
    pass

@to_start.register
def _(prices: pd.DataFrame) -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        result = prices.aggregate(to_start)
        if isinstance(result, pd.Series):
            return result
        raise ValueError

@to_start.register
def _(prices: pd.Series) -> pd.Timestamp:
    start = prices.dropna().index[0]
    if isinstance(start, pd.Timestamp):
        return start
    return pd.to_datetime(str(start))



import yfinance as yf


p = yf.download("SPY")

d = to_start(p)

print(d)