from src.core import strategy
from src.core.analytics import metrics, estimators
import yfinance as yf

prices = yf.download("ACWI, BND, VNQ, SPY, QQQ")["Adj Close"]

from src.core import Optimizer

stra = strategy.HierarchicalRiskParity(prices=prices).simulate(
    start="2015-1-1"
)
stra.value.plot()
# weights_bm = prices.notna().divide(prices.notna().sum(axis=1), axis=0)
# metrics.to_pri_return(prices=prices).multiply(weights_bm).sum(axis=1).loc["2015-1-1":].add(1).cumprod().multiply(1000).plot()
