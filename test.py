import pandas as pd
import yfinance as yf
from typing import overload, Union


@overload
def to_pri_return(prices: pd.Series) -> pd.Series:
    ...


@overload
def to_pri_return(prices: pd.DataFrame) -> pd.DataFrame:
    ...


def to_pri_return(
    prices: Union[pd.DataFrame, pd.Series]
) -> Union[pd.DataFrame, pd.Series]:
    return prices.pct_change().fillna(0)


prices = yf.download("SPY")


to_pri_return(prices=prices)
