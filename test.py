from typing import Optional
import pandas as pd
from src.core.strategies import Strategy
from src.core.analytics.features import momentum
from src.core.analytics.metrics import to_ann_volatility


class DualMomentum(Strategy):
    # Objective: balanced growth
    # Type: momentum strategy
    # Invests in: ETFs tracking stocks, bonds, real estate, and gold
    # Rebalancing schedule: monthly
    # Taxation: 50% short-term capital gains
    # Minimum account size: $5,000

    def rebalance(self) -> Optional[pd.Series]:
        safe = to_ann_volatility(self.reb_prices.iloc[-252:]).idxmin()
        mome = momentum(self.reb_prices, months=1).iloc[-1]
        safe_mome = mome.loc[safe]
        weights = {}
        for asset in self.reb_prices:
            if asset == safe:
                continue
            if mome.loc[asset] > safe_mome:
                weights.update({asset: 0.20})
            else:
                if safe in weights.keys():
                    weights.update({safe: weights[safe] + 0.20})
                else:
                    weights.update({safe: 0.20})
        print(weights)
        return pd.Series(weights)


import yfinance as yf


prices = yf.download("SPY, BND, VNQ, VCSH, GLD, BIL")["Adj Close"].dropna()

strategy = DualMomentum(prices=prices).simulate(start="2020-1-1")

strategy.value.plot()
