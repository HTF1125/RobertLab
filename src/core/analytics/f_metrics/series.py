from typing import Optional
import numpy as np
import pandas as pd


class Vectorizer:
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        if 'price' in kwargs:
            if not isinstance(kwargs['price'], pd.DataFrame):
                raise TypeError("'price' parameter must be a Pandas DataFrame")
        return self.func(*args, **kwargs)


def to_start(price: pd.Series) -> pd.Timestamp:
    """calculate start date of the price"""
    return price.dropna().index[0]


def to_end(price: pd.Series) -> pd.Timestamp:
    """calculate end date of the price"""
    return price.dropna().index[-1]


def to_pri_return(price: pd.Series) -> pd.Series:
    """calculate price return series"""
    return price.dropna().pct_change().fillna(0)


def to_log_return(price: pd.Series) -> pd.Series:
    """calculate price return series"""
    return to_pri_return(price=price).apply(np.log1p)


def to_num_year(price: pd.Series) -> pd.Series:
    """calculate num of year for price series"""
    return (to_end(price=price) - to_start(price=price)).days / 365.0


def to_num_bar(price: pd.Series) -> pd.Series:
    """calculate num of bar of price series"""
    return len(price.dropna())


def to_ann_factor(price: pd.Series) -> float:
    """calculate annualization factor"""
    return to_num_bar(price=price) / to_num_year(price=price)


def to_cum_return(price: pd.Series) -> float:
    """calculate cumulative return"""
    return price.dropna().loc[-1] / price.dropna().loc[0] - 1


def to_ann_return(price: pd.Series) -> float:
    """calculate annualized return"""
    return to_cum_return(price=price) ** (1 / to_num_year(price=price)) - 1


def to_ann_variance(price: pd.Series, ann_factor: Optional[float] = None) -> float:
    """calculate annualized variance"""
    if not ann_factor:
        ann_factor = to_ann_factor(price=price)
    return to_pri_return(price=price).var() * ann_factor


def to_ann_volatility(price: pd.Series, ann_factor: Optional[float] = None) -> float:
    """calculate annualized volatility"""
    if not ann_factor:
        ann_factor = to_ann_factor(price=price)
    return to_ann_variance(price=price, ann_factor=ann_factor) ** 0.5


def to_ann_semi_variance(price: pd.Series, ann_factor: Optional[float] = None) -> float:
    """calculate annualized semi volatility"""
    if not ann_factor:
        ann_factor = to_ann_factor(price=price)
    pri_return = to_pri_return(price=price)
    semi_pri_return = pri_return[pri_return >= 0]
    return semi_pri_return.var() * ann_factor


def to_ann_semi_volatility(
    price: pd.Series, ann_factor: Optional[float] = None
) -> float:
    """calculate annualized semi volatility"""
    if not ann_factor:
        ann_factor = to_ann_factor(price=price)
    return to_ann_semi_variance(price=price, ann_factor=ann_factor) ** 0.5


def to_drawdown(price: pd.Series, min_periods: Optional[int] = None) -> pd.Series:
    """calculate drawdown series"""
    return price / price.expanding(min_periods=min_periods or 1).max() - 1


def to_max_drawdown(price: pd.Series, min_periods: Optional[int] = None) -> float:
    """calculate maximum drawdown"""
    return to_drawdown(price=price, min_periods=min_periods).min()


def to_sharpe_ratio(
    price: pd.Series, risk_free: float = 0.0, ann_factor: Optional[float] = None
) -> float:
    """calculate sharpe ratio"""
    if not ann_factor:
        ann_factor = to_ann_factor(price=price)
    excess_return = to_ann_return(price=price) - risk_free
    return excess_return / to_ann_volatility(price=price, ann_factor=ann_factor)


def to_sortino_ratio(price: pd.Series, ann_factor: Optional[float] = None) -> float:
    """calculate sortino ratio"""
    if not ann_factor:
        ann_factor = to_ann_factor(price=price)
    return to_ann_return(price=price) / to_ann_semi_volatility(price=price)


def to_tail_ratio(price: pd.Series, alpha: float = 0.05) -> float:
    """calculate tail ratio"""
    return price.dropna().quantile(q=alpha) / price.dropna().quantile(q=1 - alpha)


def to_skewness(price: pd.Series) -> float:
    """calculate skewness"""
    return to_pri_return(price=price).skew()


def to_kurtosis(price: pd.Series) -> float:
    """calculate kurtosis"""
    return to_pri_return(price=price).kurt()


def to_value_at_risk(price: pd.Series, alpha: float = 0.05) -> float:
    """calculate value at risk"""
    return to_pri_return(price=price).quantile(q=alpha)


def to_conditional_value_at_risk(price: pd.Series, alpha: float = 0.05) -> float:
    """calculate conditional value at risk"""
    pri_return = to_pri_return(price=price)
    var = pri_return.quantile(q=alpha)
    return pri_return[pri_return < var].mean()
