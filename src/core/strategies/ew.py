"""ROBERT"""
from typing import Optional, Callable
from functools import partial
import pandas as pd
from .base import Strategy
from ..portfolios import Optimizer


def backtest(func: Callable):
    """strategy wrapper"""
    def wrapper(prices: pd.DataFrame, start: Optional[str] = None, **kwargs):
        return Strategy(
            prices=prices, rebalance=partial(func, **kwargs)
        ).simulate(start=start)
    return wrapper


@backtest
def EW(strategy: Strategy) -> Optional[pd.Series]:
    """equal"""
    return Optimizer.from_prices(prices=strategy.reb_prices).uniform_allocation()

@backtest
def HRP(strategy: Strategy) -> Optional[pd.Series]:
    """hrp"""
    return Optimizer.from_prices(prices=strategy.reb_prices).hierarchical_risk_parity()

@backtest
def HERC(strategy: Strategy) -> Optional[pd.Series]:
    """hrp"""
    return Optimizer.from_prices(prices=strategy.reb_prices).hierarchical_risk_parity()


@backtest
def RP(strategy: Strategy) -> Optional[pd.Series]:
    """hrp"""
    return Optimizer.from_prices(prices=strategy.reb_prices).risk_parity()

@backtest
def MaxSharpe(strategy: Strategy) -> Optional[pd.Series]:
    """hrp"""
    return Optimizer.from_prices(prices=strategy.reb_prices).maximized_sharpe_ratio()

@backtest
def InvVariance(strategy: Strategy) -> Optional[pd.Series]:
    """hrp"""
    return Optimizer.from_prices(prices=strategy.reb_prices).inverse_variance()


@backtest
def MeanReversion(strategy: Strategy) -> Optional[pd.Series]:
    """
    What is Mean Reversion?
        According to Investopedia, mean reversion, or reversion to the mean, is
        a theory used in finance (rooted in a concept well known as regression
        towards the mean) that suggests that asset price volatility and
        historical returns eventually will revert to the long-run mean or
        average level of the entire dataset. Mean is the average price and
        reversion means to return to, so mean reversion means “return to the
        average price”.

        While an assets price tends to revert to the average over time, this
        does not always mean or guarantee that the price will go back to the
        mean, nor does it mean that the price will rise to the mean.

    What Is A Mean Reversion Trading Strategy ?
        A mean reversion trading strategy is a trading strategy that focuses on
        when a security moves too far away from some average. The theory is that
        the price will move back toward that average at some point in time.
        There are many different ways to look at this strategy, for example by
        using linear regression, RSI, Bollinger Bands, standard deviation,
        moving averages etc. The question is how far from the average / mean is
        too far ?
    """
    return Optimizer.from_prices(prices=strategy.reb_prices).inverse_variance()



@backtest
def TargetVol(strategy: Strategy, target_vol: float = 0.10) -> Optional[pd.Series]:
    """target_vol"""
    return Optimizer.from_prices(
        prices=strategy.reb_prices,
        min_volatility=target_vol,
        max_volatility=target_vol,
    ).minimized_volatility()


@backtest
def DualMomentum(strategy: Strategy) -> Optional[pd.Series]:
    """

    Procedure:
    1. Detect market-regime by combining economic indicators with S&P 500's momentum and volatility.
    2. In bear markets, use a momentum strategy trading various debt instruments.
    3. In bull markets, use two momentum strategies: one trading sector ETFs, the other trading style-box ETFs
    * For each momentum strategy
    * create a benchmark through equal-weighting the traded assets determine
    * momentum relative to this benchmark use walk-forward optimization to
    * continually adjust the momentum filters

    """
    return Optimizer.from_prices(
        prices=strategy.reb_prices,
    ).uniform_allocation()

@backtest
def TrendFollowing(strategy: Strategy) -> Optional[pd.Series]:
    """
    The idea behind market trends is simple: Stocks that have been rising (or falling)
    in the past tend to continue growing (or falling) in the future.

    There are many methods of determining market trends.
    Among the most widely used ones are trend filters based on moving averages.
    For example, we can consider an asset to be trending up when trading above its 200-day moving average.
    It is important to note that trend is binary. Either an asset is trending or not.
    Alternatively, we could also say that assets are either trending up or trending down.
    The resulting strategies typically also have binary behavior, switching from risk-on to risk-off allocations.
    An excellent example of a trend-following strategy is Keller's Lethargic Asset Allocation.

    Link: https://www.turingtrader.com/portfolios/keller-lethargic-asset-allocation/

    """
    return Optimizer.from_prices(
        prices=strategy.reb_prices,
    ).uniform_allocation()