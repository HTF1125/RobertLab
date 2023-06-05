"""ROBERT"""
from typing import Union, List, Set, Tuple, Dict
from datetime import datetime
from functools import lru_cache
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
from .. import metrics


def get_prices(tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
    # create ticker list
    tickers = (
        tickers
        if isinstance(tickers, (list, set, tuple))
        else tickers.replace(",", " ").split()
    )
    out = []
    for ticker in tickers:
        price = get_price(ticker)
        if price is not None:
            out.append(price)
    return pd.concat(out, axis=1)


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


def get_oecd_us_leading_indicator() -> pd.DataFrame:
    return get_macro(tickers="USALOLITONOSTSAM")


def get_vix_index() -> pd.DataFrame:
    return get_prices(tickers="^VIX")


def get_universe() -> pd.DataFrame:
    import os
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(dirname, "universe.csv")
    # Load JSON from a file
    return pd.read_csv(filename)