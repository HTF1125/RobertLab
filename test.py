from src.core.signals import OECDUSLEIHP
from src.core.portfolios import Optimizer
from src.core.strategies import backtest
import pandas as pd
import yfinance as yf



@backtest
def SectorRotation(strategy, signal):

    selections = {
        "expansion": pd.Index(["XLC", "XLRE", "XLK", "XLI", "XLB"]),
        "recovery": pd.Index(["XLC", "XLF", "XLI", "XLY", "XLB"]),
        "slowdown": pd.Index(["XLU", "XLE", "XLP", "XLV"]),
        "contraction": pd.Index(["XLC", "XLY", "XLB", "XLK"]),
    }

    state = signal.get_state(str(strategy.date))
    index = selections.get(state)

    allocations = Optimizer.from_prices(prices=strategy.reb_prices[index]).uniform_allocation()
    return allocations


prices = yf.download(tickers = "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU")["Adj Close"]
signal = OECDUSLEIHP.from_fred_data()

strategy = SectorRotation(prices=prices, signal=signal, start="2010-1-1")
strategy.value.plot()

print(strategy.value)