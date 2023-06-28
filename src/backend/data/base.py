"""ROBERT"""

import pandas as pd
import yfinance as yf
import pandas_datareader as pdr




def get_price(ticker: str) -> pd.DataFrame:
    return yf.download(ticker, progress=False)