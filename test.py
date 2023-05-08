from app.core.portfolios import Optimizer
from app.core.strategies import Strategy

class MaxSharpe(Strategy):
    def rebalance(self, **kwargs):
        prices = self.reb_prices.iloc[-252:]
        opt = Optimizer.from_prices(prices)
        return opt.maximized_sharpe_ratio()

import yfinance as yf

prices = yf.download("SPY, VNQ, XLK, XLU, XLB, XLV, XLY, XLG, BIL, AGG, TLT")["Adj Close"]

strategy = MaxSharpe(prices=prices).simulate()

strategy.value.plot()