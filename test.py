from src.core import strategy
from src.core import metrics
import yfinance as yf

prices = yf.download("XLU, XLK, XLB, XLP, XLY, XLI, XLV, XLF")["Adj Close"]

strategy = strategy.Momentum(prices).simulate(
    start="2015-1-1", min_volatility=0.12
)

strategy.value.plot()
print(strategy.analytics())
