from typing import Optional
from src.core.strategies import Strategy
from functools import partial
import pandas as pd

data = pd.read_clipboard()
port3 = data[data.strategy == "MLP_US_3"]
port4 = data[data.strategy == "MLP_US_4"]
port5 = data[data.strategy == "MLP_US_5"]

def rebalance(strategy: Strategy, port: pd.DataFrame) -> Optional[pd.Series]:

    p = port.loc[strategy.date].dropna()
    print(p)
    return p

import yfinance as yf

cport = (
    port5[["date", "ticker_bloomberg", "weight"]]
    .set_index(["date", "ticker_bloomberg"])
    .unstack()["weight"]
)
cport.index = pd.to_datetime(cport.index)
cport = cport.sort_index()
cport.columns = [t.replace(" US Equity", "") for t in cport.columns]
cport.index = cport.index - pd.DateOffset(months=1)
cport = cport.resample("M").last()
prices = yf.download(tickers=list(cport.columns))["Adj Close"]

strategy = Strategy(prices=prices, rebalance=partial(rebalance, port=cport)).simulate(start="2017-12-28")
strategy.analytics()