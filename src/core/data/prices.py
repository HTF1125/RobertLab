import concurrent.futures
from typing import Union, List, Set, Tuple
from functools import lru_cache
import pandas as pd
import yfinance as yf


def prices(tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:

    # create ticker list
    tickers = (
        tickers
        if isinstance(tickers, (list, set, tuple))
        else tickers.replace(",", " ").split()
    )

    return pd.concat([
        price(ticker) for ticker in tickers
    ], axis=1)


@lru_cache()
def price(ticker: str) -> pd.Series:
    p = yf.download(ticker, progress=False)["Adj Close"]
    p.name = ticker
    return p
