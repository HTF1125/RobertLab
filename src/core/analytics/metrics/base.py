"""ROBERT"""
from typing import Union, Optional
from typing import overload
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from ...ext.periods import AnnFactor


@overload
def to_start(prices: pd.Series) -> pd.Timestamp:
    ...


@overload
def to_start(prices: pd.DataFrame) -> pd.Series:
    ...


def to_start(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, pd.Timestamp]:
    """get the start of the data"""
    if isinstance(prices, pd.DataFrame):
        result = prices.apply(to_start)
        if isinstance(result, pd.Series):
            return result
        raise ValueError
    start = prices.dropna().index[0]
    if isinstance(start, pd.Timestamp):
        return start
    return pd.to_datetime(str(start))


@overload
def to_end(prices: pd.Series) -> pd.Timestamp:
    ...


@overload
def to_end(prices: pd.DataFrame) -> pd.Series:
    ...


def to_end(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, pd.Timestamp]:
    """get the start of the data"""
    if isinstance(prices, pd.DataFrame):
        result = prices.apply(to_end)
        if isinstance(result, pd.Series):
            return result
        raise ValueError
    start = prices.dropna().index[-1]
    if isinstance(start, pd.Timestamp):
        return start
    return pd.to_datetime(str(start))


@overload
def to_pri_return(prices: pd.Series) -> pd.Series:
    ...


@overload
def to_pri_return(prices: pd.DataFrame) -> pd.DataFrame:
    ...


def to_pri_return(
    prices: Union[pd.DataFrame, pd.Series]
) -> Union[pd.DataFrame, pd.Series]:
    """calculate prices return series"""
    if isinstance(prices, pd.DataFrame):
        return prices.apply(to_pri_return)

    pri_return = prices.dropna().pct_change().fillna(0)
    return pri_return


@overload
def to_log_return(prices: pd.Series) -> pd.Series:
    ...


@overload
def to_log_return(prices: pd.DataFrame) -> pd.DataFrame:
    ...


def to_log_return(
    prices: Union[pd.DataFrame, pd.Series]
) -> Union[pd.DataFrame, pd.Series]:
    """calculate prices return series"""
    return to_pri_return(prices=prices).apply(np.log1p)


@overload
def to_cum_return(prices: pd.DataFrame) -> pd.DataFrame:
    ...


@overload
def to_cum_return(prices: pd.Series) -> pd.Series:
    ...


def to_cum_return(
    prices: Union[pd.DataFrame, pd.Series]
) -> Union[pd.DataFrame, pd.Series]:
    if isinstance(prices, pd.DataFrame):
        return prices.apply(to_cum_return)
    return prices.dropna().iloc[-1] / prices.dropna().iloc[0] - 1


@overload
def to_ann_return(prices: pd.DataFrame, ann_factor: Union[int, float]) -> pd.Series:
    ...


@overload
def to_ann_return(prices: pd.Series, ann_factor: Union[int, float]) -> float:
    ...


def to_ann_return(
    prices: Union[pd.DataFrame, pd.Series],
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> Union[pd.Series, float]:
    return (to_log_return(prices=prices).apply(np.exp).mean() - 1) * ann_factor


@overload
def to_ann_variance(prices: pd.DataFrame, ann_factor: Union[int, float]) -> pd.Series:
    ...


@overload
def to_ann_variance(prices: pd.Series, ann_factor: Union[int, float]) -> float:
    ...


def to_ann_variance(
    prices: Union[pd.DataFrame, pd.Series],
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> Union[pd.Series, float]:
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_ann_variance, ann_factor=ann_factor)
    return float(np.var(to_pri_return(prices=prices)) * ann_factor)


@overload
def to_ann_volatility(prices: pd.DataFrame, ann_factor: Union[int, float]) -> pd.Series:
    ...


@overload
def to_ann_volatility(prices: pd.Series, ann_factor: Union[int, float]) -> float:
    ...


def to_ann_volatility(
    prices: Union[pd.DataFrame, pd.Series],
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> Union[pd.Series, float]:
    return to_ann_variance(prices=prices, ann_factor=ann_factor) ** 0.5


@overload
def to_ann_semi_variance(
    prices: pd.DataFrame, ann_factor: Union[int, float]
) -> pd.Series:
    ...


@overload
def to_ann_semi_variance(prices: pd.Series, ann_factor: Union[int, float]) -> float:
    ...


def to_ann_semi_variance(
    prices: Union[pd.DataFrame, pd.Series],
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> Union[pd.Series, float]:
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    pri_return = to_pri_return(prices=prices)
    semi_var = pri_return[pri_return >= 0].var()
    out = semi_var * ann_factor
    if isinstance(prices, pd.Series):
        return out.iloc[0]
    return out


@overload
def to_ann_semi_volatility(
    prices: pd.DataFrame, ann_factor: Union[int, float]
) -> pd.Series:
    ...


@overload
def to_ann_semi_volatility(prices: pd.Series, ann_factor: Union[int, float]) -> float:
    ...


def to_ann_semi_volatility(
    prices: Union[pd.DataFrame, pd.Series],
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> Union[pd.Series, float]:
    return to_ann_semi_variance(prices=prices, ann_factor=ann_factor) ** 0.5


@overload
def to_drawdown(prices: pd.DataFrame, min_periods: int) -> pd.DataFrame:
    ...


@overload
def to_drawdown(prices: pd.Series, min_periods: int) -> pd.Series:
    ...


def to_drawdown(
    prices: Union[pd.DataFrame, pd.Series],
    min_periods: int = 0,
) -> Union[pd.DataFrame, pd.Series]:
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_drawdown, min_periods=min_periods)
    return prices / prices.expanding(min_periods=min_periods or 1).max() - 1


@overload
def to_max_drawdown(prices: pd.DataFrame) -> pd.Series:
    ...


@overload
def to_max_drawdown(prices: pd.Series) -> float:
    ...


def to_max_drawdown(
    prices: Union[pd.DataFrame, pd.Series],
) -> Union[pd.Series, float]:
    return to_drawdown(prices=prices, min_periods=0).min()


@overload
def to_sharpe_ratio(
    prices: pd.DataFrame, risk_free: Union[int, float], ann_factor: Union[int, float]
) -> pd.Series:
    ...


@overload
def to_sharpe_ratio(
    prices: pd.Series, risk_free: Union[int, float], ann_factor: Union[int, float]
) -> float:
    ...


def to_sharpe_ratio(
    prices: Union[pd.DataFrame, pd.Series],
    risk_free: Union[int, float] = 0.0,
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> Union[pd.Series, float]:
    excess_return = to_ann_return(prices=prices, ann_factor=ann_factor) - risk_free
    return excess_return / to_ann_volatility(prices=prices, ann_factor=ann_factor)


@overload
def to_sortino_ratio(prices: pd.DataFrame, ann_factor: Union[int, float]) -> pd.Series:
    ...


@overload
def to_sortino_ratio(prices: pd.Series, ann_factor: Union[int, float]) -> float:
    ...


def to_sortino_ratio(
    prices: Union[pd.DataFrame, pd.Series],
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> Union[pd.Series, float]:
    return to_ann_return(prices=prices, ann_factor=ann_factor) / to_ann_semi_volatility(
        prices=prices, ann_factor=ann_factor
    )


@overload
def to_tail_ratio(prices: pd.DataFrame, alpha: float) -> pd.Series:
    ...


@overload
def to_tail_ratio(prices: pd.Series, alpha: float) -> float:
    ...


def to_tail_ratio(
    prices: Union[pd.DataFrame, pd.Series], alpha: float = 0.05
) -> Union[pd.Series, float]:
    return prices.dropna().quantile(q=alpha) / prices.dropna().quantile(q=1 - alpha)


@overload
def to_skewness(prices: pd.DataFrame, log_return: bool) -> pd.Series:
    ...


@overload
def to_skewness(prices: pd.Series, log_return: bool) -> float:
    ...


def to_skewness(
    prices: Union[pd.DataFrame, pd.Series], log_return: bool = False
) -> Union[pd.Series, float]:
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_skewness, log_return=log_return)
    if log_return:
        pri_return = to_log_return(prices=prices)
    pri_return = to_pri_return(prices=prices)
    n = len(pri_return)
    mean = pri_return.mean()
    std = pri_return.std()
    skewness = (1 / n) * ((pri_return - mean) / std).pow(3).sum()
    return float(skewness)


@overload
def to_kurtosis(prices: pd.DataFrame, log_return: bool) -> pd.Series:
    ...


@overload
def to_kurtosis(prices: pd.Series, log_return: bool) -> float:
    ...


def to_kurtosis(
    prices: Union[pd.DataFrame, pd.Series], log_return: bool = False
) -> Union[pd.Series, float]:
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_kurtosis, log_return=log_return)
    if log_return:
        pri_return = to_log_return(prices=prices)
    pri_return = to_pri_return(prices=prices)
    n = len(pri_return)
    mean = pri_return.mean()
    std = pri_return.std()
    kurtosis = (1 / n) * ((pri_return - mean) / std).pow(4).sum()
    return float(kurtosis)


@overload
def to_value_at_risk(prices: pd.DataFrame, alpha: float) -> pd.Series:
    ...


@overload
def to_value_at_risk(prices: pd.Series, alpha: float) -> float:
    ...


def to_value_at_risk(
    prices: Union[pd.DataFrame, pd.Series],
    alpha: float = 0.05,
) -> Union[pd.Series, float]:
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_value_at_risk, alpha=alpha)
    return to_pri_return(prices=prices).quantile(q=alpha)


@overload
def to_conditional_value_at_risk(prices: pd.DataFrame, alpha: float) -> pd.Series:
    ...


@overload
def to_conditional_value_at_risk(prices: pd.Series, alpha: float) -> float:
    ...


def to_conditional_value_at_risk(
    prices: Union[pd.DataFrame, pd.Series],
    alpha: float = 0.05,
) -> Union[pd.Series, float]:
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_conditional_value_at_risk, alpha=alpha)
    pri_return = to_pri_return(prices=prices)
    var = pri_return.quantile(q=alpha)
    return pri_return[pri_return < var].mean()


@overload
def momentum(prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
    ...


@overload
def momentum(prices: pd.Series, **kwargs) -> pd.Series:
    ...


def momentum(
    prices: Union[pd.DataFrame, pd.Series], **kwargs
) -> Union[pd.DataFrame, pd.Series]:
    resampled_prices = prices.resample("D").last().ffill()
    offset_prices = resampled_prices.shift(1, freq=pd.DateOffset(**kwargs))
    return (prices / offset_prices).loc[prices.index]


def to_expected_returns(prices: pd.DataFrame) -> pd.Series:
    """Calculates the expected returns from a DataFrame of prices.

    Args:
        prices (pd.DataFrame): A DataFrame of asset prices.

    Returns:
        pd.Series: A Series of expected returns.
    """
    return to_ann_return(prices=prices)


def exponential_alpha(
    com: Optional[float] = None,
    span: Optional[float] = None,
    halflife: Optional[float] = None,
) -> float:
    """_summary_

    Args:
        com (Optional[float], optional): _description_. Defaults to None.
        span (Optional[float], optional): _description_. Defaults to None.
        halflife (Optional[float], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        float: _description_
    """
    if com:
        return 1 / (1 + com)
    if span:
        return 2 / (span + 1)
    if halflife:
        return 1 - np.exp(-np.log(2) / halflife)

    return 0.0


def to_covariance_matrix(
    prices: pd.DataFrame,
    ann_factor: float = AnnFactor.daily,
    com: Optional[float] = None,
    span: Optional[float] = None,
    halflife: Optional[float] = None,
) -> pd.DataFrame:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        ann_factor (Optional[Union[int, float, pd.Series]], optional): _description_. Defaults to None.
        com (Optional[float], optional): _description_. Defaults to None.
        span (Optional[float], optional): _description_. Defaults to None.
        halflife (Optional[float], optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """

    pri_returns = to_pri_return(prices=prices)

    if com is not None or span is not None or halflife is not None:
        alpha = exponential_alpha(com=com, span=span, halflife=halflife)

        exp_covariance_matrix = (
            pri_returns.ewm(alpha=alpha).cov().unstack().iloc[-1].unstack() * ann_factor
        )

        return exp_covariance_matrix.loc[prices.columns, prices.columns]
    return pri_returns.cov() * ann_factor


def to_correlation_matrix(
    prices: pd.DataFrame,
    com: Optional[float] = None,
    span: Optional[float] = None,
    halflife: Optional[float] = None,
) -> pd.DataFrame:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        ann_factor (Optional[Union[int, float, pd.Series]], optional): _description_. Defaults to None.
        com (Optional[float], optional): _description_. Defaults to None.
        span (Optional[float], optional): _description_. Defaults to None.
        halflife (Optional[float], optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """

    pri_returns = to_pri_return(prices=prices)

    if com is not None or span is not None or halflife is not None:
        alpha = exponential_alpha(com=com, span=span, halflife=halflife)

        exp_covariance_matrix = (
            pri_returns.ewm(alpha=alpha).corr().unstack().iloc[-1].unstack()
        )

        return exp_covariance_matrix.loc[prices.columns, prices.columns]
    return pri_returns.corr()
