from typing import Optional, Callable
import numpy as np
import pandas as pd
from ...ext.periods import AnnFactor


def vectorize(func) -> Callable:
    """apply vectorized prices"""

    def wrapper(*args, **kwargs):
        if args:
            prices = args[0]
            args = args[1:]
        else:
            prices = kwargs.get("prices")
            del kwargs["prices"]
        if isinstance(prices, pd.DataFrame):
            out = prices.apply(func, *args, **kwargs)
            out.name = func.__name__.replace("to_", "")
            return out
        elif isinstance(prices, pd.Series):
            return func(*args, prices=prices, **kwargs)
        else:
            raise TypeError("prices must be pd.Series or pd.DataFrame.")

    return wrapper


@vectorize
def to_start(prices: pd.Series) -> pd.Timestamp:
    """calculate start date of the prices"""

    start = prices.dropna().index[0]
    if isinstance(start, pd.Timestamp):
        return start
    return pd.to_datetime(str(start))


@vectorize
def to_end(prices: pd.Series) -> pd.Timestamp:
    """calculate end date of the prices"""
    end = prices.dropna().index[-1]
    if isinstance(end, pd.Timestamp):
        return end
    return pd.to_datetime(str(end))


@vectorize
def to_pri_return(prices: pd.Series) -> pd.Series:
    """calculate prices return series"""
    return prices.dropna().pct_change().fillna(0)


@vectorize
def to_log_return(prices: pd.Series) -> pd.Series:
    """calculate prices return series"""
    return to_pri_return(prices=prices).apply(np.log1p)


@vectorize
def to_num_year(prices: pd.Series) -> float:
    """calculate num of year for prices series"""
    return (to_end(prices=prices) - to_start(prices=prices)).days / 365.0


@vectorize
def to_num_bar(prices: pd.Series) -> int:
    """calculate num of bar of prices series"""
    return len(prices.dropna())


@vectorize
def to_ann_factor(prices: pd.Series) -> float:
    """calculate annualization factor"""
    return to_num_bar(prices=prices) / to_num_year(prices=prices)


@vectorize
def to_cum_return(prices: pd.Series) -> float:
    """calculate cumulative return"""
    return prices.dropna().iloc[-1] / prices.dropna().iloc[0] - 1


@vectorize
def to_ann_return(prices: pd.Series, ann_factor: float = AnnFactor.daily) -> float:
    """calculate annualized return"""
    return np.exp(to_log_return(prices=prices).mean() * ann_factor) - 1


@vectorize
def to_ann_variance(prices: pd.Series, ann_factor: float = AnnFactor.daily) -> float:
    """calculate annualized variance"""
    return to_pri_return(prices=prices).var() * ann_factor


@vectorize
def to_ann_volatility(prices: pd.Series, ann_factor: float = AnnFactor.daily) -> float:
    """calculate annualized volatility"""

    return to_ann_variance(prices=prices, ann_factor=ann_factor) ** 0.5


@vectorize
def to_ann_semi_variance(
    prices: pd.Series, ann_factor: float = AnnFactor.daily
) -> float:
    """calculate annualized semi volatility"""

    pri_return = to_pri_return(prices=prices)
    semi_pri_return = pri_return[pri_return >= 0]
    return semi_pri_return.var() * ann_factor


@vectorize
def to_ann_semi_volatility(
    prices: pd.Series, ann_factor: float = AnnFactor.daily
) -> float:
    """calculate annualized semi volatility"""

    return to_ann_semi_variance(prices=prices, ann_factor=ann_factor) ** 0.5


@vectorize
def to_drawdown(prices: pd.Series, min_periods: Optional[int] = None) -> pd.Series:
    """calculate drawdown series"""
    return prices / prices.expanding(min_periods=min_periods or 1).max() - 1


@vectorize
def to_max_drawdown(prices: pd.Series, min_periods: Optional[int] = None) -> float:
    """calculate maximum drawdown"""
    return to_drawdown(prices=prices, min_periods=min_periods).min()


@vectorize
def to_sharpe_ratio(
    prices: pd.Series, risk_free: float = 0.0, ann_factor: float = AnnFactor.daily
) -> float:
    """calculate sharpe ratio"""

    excess_return = to_ann_return(prices=prices) - risk_free
    return excess_return / to_ann_volatility(prices=prices, ann_factor=ann_factor)


@vectorize
def to_sortino_ratio(prices: pd.Series, ann_factor: float = AnnFactor.daily) -> float:
    """calculate sortino ratio"""

    return to_ann_return(prices=prices, ann_factor=ann_factor) / to_ann_semi_volatility(
        prices=prices, ann_factor=ann_factor
    )


@vectorize
def to_tail_ratio(prices: pd.Series, alpha: float = 0.05) -> float:
    """calculate tail ratio"""
    return prices.dropna().quantile(q=alpha) / prices.dropna().quantile(q=1 - alpha)


@vectorize
def to_skewness(prices: pd.Series, log_return: bool = False) -> float:
    """calculate skewness"""
    if log_return:
        return to_log_return(prices=prices).skew()
    return to_pri_return(prices=prices).skew()


@vectorize
def to_kurtosis(prices: pd.Series, log_return: bool = False) -> float:
    """calculate kurtosis"""
    if log_return:
        return to_log_return(prices=prices).kurt()
    return to_pri_return(prices=prices).kurt()


@vectorize
def to_value_at_risk(prices: pd.Series, alpha: float = 0.05) -> float:
    """calculate value at risk"""
    return to_pri_return(prices=prices).quantile(q=alpha)


@vectorize
def to_conditional_value_at_risk(prices: pd.Series, alpha: float = 0.05) -> float:
    """calculate conditional value at risk"""
    pri_return = to_pri_return(prices=prices)
    var = pri_return.quantile(q=alpha)
    return pri_return[pri_return < var].mean()
