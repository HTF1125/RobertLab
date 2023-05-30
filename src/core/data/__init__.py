"""ROBERT"""
from typing import Union, List, Set, Tuple
from datetime import datetime
from functools import lru_cache
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr


def get_prices(tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
    if isinstance(tickers, dict):
        out = get_prices(tickers=list(tickers.keys()))
        out = out.rename(columns=tickers)
        return out
    # create ticker list
    tickers = (
        tickers
        if isinstance(tickers, (list, set, tuple))
        else tickers.replace(",", " ").split()
    )

    out = pd.concat([get_price(ticker) for ticker in tickers], axis=1)
    return out


@lru_cache()
def get_price(ticker: str) -> pd.Series:
    p = yf.download(ticker, progress=False)["Adj Close"]
    p.name = ticker
    return p


@lru_cache()
def fred_data(ticker: str) -> pd.Series:
    """get the fred data"""
    out = pdr.DataReader(
        name=ticker,
        data_source="fred",
        start=datetime(1900, 1, 1),
    )
    if isinstance(out, pd.DataFrame):
        out = out.squeeze()
        out.name = ticker

    return out


def get_macro(tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
    # create ticker list
    tickers = (
        tickers
        if isinstance(tickers, (list, set, tuple))
        else tickers.replace(",", " ").split()
    )
    return pd.concat([fred_data(ticker) for ticker in tickers], axis=1)

