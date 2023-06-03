"""ROBERT"""
import warnings
from typing import Union, Optional, List
from typing import overload
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from ..ext.periods import AnnFactor


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

    pri_return = prices.dropna().pct_change().iloc[1:]
    return pri_return


@overload
def to_log_return(prices: pd.Series) -> pd.Series:
    ...


@overload
def to_log_return(prices: pd.DataFrame) -> pd.DataFrame:
    ...


@overload
def to_log_return(
    prices: Union[pd.DataFrame, pd.Series]
) -> Union[pd.DataFrame, pd.Series]:
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
def to_ann_return(
    prices: pd.DataFrame, ann_factor: Union[int, float] = AnnFactor.daily
) -> pd.Series:
    ...


@overload
def to_ann_return(
    prices: pd.Series, ann_factor: Union[int, float] = AnnFactor.daily
) -> float:
    ...


def to_ann_return(
    prices: Union[pd.DataFrame, pd.Series],
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> Union[pd.Series, float]:
    return (to_log_return(prices=prices).apply(np.exp).mean() - 1) * ann_factor


@overload
def to_ann_variance(
    prices: pd.DataFrame, ann_factor: Union[int, float] = AnnFactor.daily
) -> float:
    ...


@overload
def to_ann_variance(
    prices: pd.Series, ann_factor: Union[int, float] = AnnFactor.daily
) -> float:
    ...


def to_ann_variance(
    prices: Union[pd.DataFrame, pd.Series],
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> Union[pd.Series, float]:
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_ann_variance, ann_factor=ann_factor)
    return float(np.var(to_pri_return(prices=prices)) * ann_factor)


@overload
def to_ann_volatility(
    prices: pd.Series, ann_factor: Union[int, float] = AnnFactor.daily
) -> float:
    ...


@overload
def to_ann_volatility(
    prices: pd.DataFrame, ann_factor: Union[int, float] = AnnFactor.daily
) -> pd.Series:
    ...


def to_ann_volatility(
    prices: Union[pd.DataFrame, pd.Series],
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> Union[pd.Series, float]:
    return to_ann_variance(prices=prices, ann_factor=ann_factor) ** 0.5


@overload
def to_ann_semi_variance(prices: pd.DataFrame) -> pd.Series:
    ...


@overload
def to_ann_semi_variance(prices: pd.Series) -> float:
    ...


@overload
def to_ann_semi_variance(
    prices: pd.DataFrame,
    ann_factor: Union[int, float] = AnnFactor.daily,
    threshold: float = 0.0,
) -> pd.Series:
    ...


@overload
def to_ann_semi_variance(
    prices: pd.Series,
    ann_factor: Union[int, float] = AnnFactor.daily,
    threshold: float = 0.0,
) -> float:
    ...


def to_ann_semi_variance(
    prices: Union[pd.DataFrame, pd.Series],
    ann_factor: Union[int, float] = AnnFactor.daily,
    threshold: float = 0.0,
) -> Union[pd.Series, float]:
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_ann_semi_variance, ann_factor=ann_factor)
    pri_return = to_pri_return(prices=prices)
    return float(np.var(pri_return[pri_return < threshold])) * ann_factor


@overload
def to_ann_semi_volatility(
    prices: pd.DataFrame,
    ann_factor: Union[int, float] = AnnFactor.daily,
    threshold: float = 0.0,
) -> pd.Series:
    ...


@overload
def to_ann_semi_volatility(
    prices: pd.Series,
    ann_factor: Union[int, float] = AnnFactor.daily,
    threshold: float = 0.0,
) -> float:
    ...


def to_ann_semi_volatility(
    prices: Union[pd.DataFrame, pd.Series],
    ann_factor: Union[int, float] = AnnFactor.daily,
    threshold: float = 0.0,
) -> Union[pd.Series, float]:
    return (
        to_ann_semi_variance(prices=prices, ann_factor=ann_factor, threshold=threshold)
        ** 0.5
    )


@overload
def to_drawdown(prices: pd.DataFrame, min_periods: int = 0) -> pd.DataFrame:
    ...


@overload
def to_drawdown(prices: pd.Series, min_periods: int = 0) -> pd.Series:
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
    prices: pd.DataFrame,
    risk_free: Union[int, float] = 0.0,
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> pd.Series:
    ...


@overload
def to_sharpe_ratio(
    prices: pd.Series,
    risk_free: Union[int, float] = 0.0,
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> float:
    ...


def to_sharpe_ratio(
    prices: Union[pd.DataFrame, pd.Series],
    risk_free: Union[int, float] = 0.0,
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> Union[pd.Series, float]:
    """
    Calculates the Sharpe ratio for a given set of prices.

    The Sharpe ratio is a measure of risk-adjusted return, indicating the excess return of an investment
    per unit of risk. It is commonly used to evaluate the performance of investment strategies.

    Args:
        prices (Union[pd.DataFrame, pd.Series]): The price data used to calculate returns.
            It can be either a DataFrame with multiple price series or a single Series.
        risk_free (Union[int, float], optional): The risk-free rate of return. Default is 0.0.
        ann_factor (Union[int, float], optional): The annualization factor for returns and volatility.
            It determines the frequency of returns used in the calculation. Default is AnnFactor.daily.

    Returns:
        Union[pd.Series, float]: The Sharpe ratio as a single value or a Series of Sharpe ratios
        if multiple price series are provided.

    Limitations:
        - The Sharpe ratio assumes the normal distribution of returns.
        - It has a bias toward high-frequency trading strategies, favoring strategies that generate
          small frequent profits and assuming proportional scaling of profits, which may not hold true.
        - It does not account for tail risk.

    """
    excess_return = to_ann_return(prices=prices, ann_factor=ann_factor) - risk_free
    return excess_return / to_ann_volatility(prices=prices, ann_factor=ann_factor)


@overload
def to_sortino_ratio(
    prices: pd.DataFrame, ann_factor: Union[int, float] = AnnFactor.daily
) -> pd.Series:
    ...


@overload
def to_sortino_ratio(
    prices: pd.Series, ann_factor: Union[int, float] = AnnFactor.daily
) -> float:
    ...


def to_sortino_ratio(
    prices: Union[pd.DataFrame, pd.Series],
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> Union[pd.Series, float]:
    """
    Calculates the Sortino ratio for a given set of prices.

    The Sortino ratio is a measure of risk-adjusted return that considers only the downside volatility
    of an investment. It is similar to the Sharpe ratio, but it focuses on the negative deviations
    from the desired return, providing a better assessment of the risk associated with an investment.

    Args:
        prices (Union[pd.DataFrame, pd.Series]): The price data used to calculate returns.
            It can be either a DataFrame with multiple price series or a single Series.
        ann_factor (Union[int, float], optional): The annualization factor for returns and volatility.
            It determines the frequency of returns used in the calculation. Default is AnnFactor.daily.

    Returns:
        Union[pd.Series, float]: The Sortino ratio as a single value or a Series of Sortino ratios
        if multiple price series are provided.

    """
    return to_ann_return(prices=prices, ann_factor=ann_factor) / to_ann_semi_volatility(
        prices=prices, ann_factor=ann_factor
    )


@overload
def to_tail_ratio(prices: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    ...


@overload
def to_tail_ratio(prices: pd.Series, alpha: float = 0.05) -> float:
    ...


def to_tail_ratio(
    prices: Union[pd.DataFrame, pd.Series], alpha: float = 0.05
) -> Union[pd.Series, float]:
    """
    Calculates the tail ratio for a given set of prices.

    The tail ratio is a measure that compares the returns in the lower tail (alpha percentile) to
    the returns in the upper tail (1 - alpha percentile) of a distribution. It provides an indication
    of the asymmetry or skewness of returns.

    Args:
        prices (Union[pd.DataFrame, pd.Series]): The price data used to calculate the tail ratio.
            It can be either a DataFrame with multiple price series or a single Series.
        alpha (float, optional): The significance level used to calculate the quantiles.
            Default is 0.05, indicating a 5% significance level.

    Returns:
        Union[pd.Series, float]: The tail ratio as a single value or a Series of tail ratios
        if multiple price series are provided.

    """
    return prices.dropna().quantile(q=alpha) / prices.dropna().quantile(q=1 - alpha)


@overload
def to_skewness(prices: pd.DataFrame, log_return: bool = False) -> pd.Series:
    ...


@overload
def to_skewness(prices: pd.Series, log_return: bool = False) -> float:
    ...


def to_skewness(
    prices: Union[pd.DataFrame, pd.Series], log_return: bool = False
) -> Union[pd.Series, float]:
    """
    Calculates the skewness of returns for a given set of prices.

    Skewness is a measure of the asymmetry or lack of symmetry in the distribution of returns.
    It indicates whether the returns are concentrated on one side of the mean, either to the left
    (negative skewness) or to the right (positive skewness).

    Args:
        prices (Union[pd.DataFrame, pd.Series]): The price data used to calculate returns.
            It can be either a DataFrame with multiple price series or a single Series.
        log_return (bool, optional): Specifies whether to use logarithmic returns.
            If True, logarithmic returns are calculated. If False, simple returns are calculated.
            Default is False.

    Returns:
        Union[pd.Series, float]: The skewness of returns as a single value or a Series of skewness
        if multiple price series are provided.

    """
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_skewness, log_return=log_return)

    if log_return:
        pri_return = to_log_return(prices=prices)
    else:
        pri_return = to_pri_return(prices=prices)

    n = len(pri_return)
    mean = pri_return.mean()
    std = pri_return.std()
    skewness = (1 / n) * ((pri_return - mean) / std).pow(3).sum()

    return float(skewness)


@overload
def to_kurtosis(prices: pd.DataFrame, log_return: bool = False) -> pd.Series:
    ...


@overload
def to_kurtosis(prices: pd.Series, log_return: bool = False) -> float:
    ...


def to_kurtosis(
    prices: Union[pd.DataFrame, pd.Series], log_return: bool = False
) -> Union[pd.Series, float]:
    """
    Calculates the kurtosis of returns for a given set of prices.

    Kurtosis is a measure of the "tailedness" or the degree of outliers in the distribution of returns.
    It indicates the presence of fat tails or extreme observations in the data.

    Args:
        prices (Union[pd.DataFrame, pd.Series]): The price data used to calculate returns.
            It can be either a DataFrame with multiple price series or a single Series.
        log_return (bool, optional): Specifies whether to use logarithmic returns.
            If True, logarithmic returns are calculated. If False, simple returns are calculated.
            Default is False.

    Returns:
        Union[pd.Series, float]: The kurtosis of returns as a single value or a Series of kurtosis
        if multiple price series are provided.

    """
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_kurtosis, log_return=log_return)

    if log_return:
        pri_return = to_log_return(prices=prices)
    else:
        pri_return = to_pri_return(prices=prices)

    n = len(pri_return)
    mean = pri_return.mean()
    std = pri_return.std()
    kurtosis = (1 / n) * ((pri_return - mean) / std).pow(4).sum()

    return float(kurtosis)


@overload
def to_value_at_risk(prices: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    ...


@overload
def to_value_at_risk(prices: pd.Series, alpha: float = 0.05) -> float:
    ...


def to_value_at_risk(
    prices: Union[pd.DataFrame, pd.Series],
    alpha: float = 0.05,
) -> Union[pd.Series, float]:
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_value_at_risk, alpha=alpha)
    return to_pri_return(prices=prices).quantile(q=alpha)


@overload
def to_conditional_value_at_risk(
    prices: pd.DataFrame, alpha: float = 0.05
) -> pd.Series:
    ...


@overload
def to_conditional_value_at_risk(prices: pd.Series, alpha: float = 0.05) -> float:
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
def to_momentum(
    prices: pd.DataFrame,
    months: int = 12,
    skip_months: int = 0,
) -> pd.Series:
    ...


@overload
def to_momentum(
    prices: pd.Series,
    months: int = 12,
    skip_months: int = 0,
) -> float:
    ...


def to_momentum(
    prices: Union[pd.DataFrame, pd.Series],
    months: int = 12,
    skip_months: int = 0,
) -> Union[pd.Series, float]:
    end = pd.Timestamp(str(prices.index[-1]))
    start = end - pd.tseries.offsets.DateOffset(months=months)
    start += pd.tseries.offsets.DateOffset(days=1)
    end -= pd.tseries.offsets.DateOffset(months=skip_months)
    pri_return = to_pri_return(prices=prices)
    if isinstance(pri_return, pd.DataFrame):
        return pri_return.loc[start:end].add(1).prod().subtract(1)
    return float(np.prod(pri_return.loc[start:end].add(1))) - 1


@overload
def to_expected_returns(prices: pd.DataFrame) -> pd.Series:
    ...


@overload
def to_expected_returns(prices: pd.Series) -> float:
    ...


def to_expected_returns(
    prices: Union[pd.DataFrame, pd.Series]
) -> Union[pd.Series, float]:
    """this is a pass through function"""
    return to_ann_return(prices=prices, ann_factor=AnnFactor.daily)


@overload
def to_monthly_return(prices: pd.DataFrame) -> pd.DataFrame:
    ...


@overload
def to_monthly_return(prices: pd.Series) -> pd.Series:
    ...


def to_monthly_return(
    prices: Union[pd.DataFrame, pd.Series]
) -> Union[pd.DataFrame, pd.Series]:
    return (
        to_pri_return(prices=prices)
        .groupby([lambda x: x.year, lambda x: x.month])
        .apply(lambda p: p.add(1).prod() - 1)
    )


@overload
def moving_average(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    ...


@overload
def moving_average(prices: pd.Series, window: int = 20) -> float:
    ...


def moving_average(
    prices: Union[pd.DataFrame, pd.Series], window: int = 20
) -> Union[pd.Series, float]:
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(moving_average, window=window)
    return prices.iloc[-window:].mean()


@overload
def to_exponential_moving_average(prices: pd.DataFrame, span: int = 20) -> pd.Series:
    ...


@overload
def to_exponential_moving_average(prices: pd.Series, span: int = 20) -> float:
    ...


def to_exponential_moving_average(
    prices: Union[pd.DataFrame, pd.Series], span: int = 20
) -> Union[pd.Series, float]:
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_exponential_moving_average, span=span)
    return prices.dropna().ewm(span=span).mean().iloc[-1]


def to_covariance_matrix(
    prices: pd.DataFrame, ann_factor: int = AnnFactor.daily, span: Optional[int] = None
) -> pd.DataFrame:
    if span:
        return to_exponential_covariance_matrix(
            prices=prices, span=span, ann_factor=ann_factor
        )
    return to_pri_return(prices=prices).cov() * ann_factor


def to_correlation_matrix(
    prices: pd.DataFrame, span: Optional[int] = None
) -> pd.DataFrame:
    if span:
        return to_exponential_correlation_matrix(prices=prices, span=span)

    return to_pri_return(prices=prices).corr()


def to_exponential_covariance_matrix(
    prices: pd.DataFrame, span: int = 180, ann_factor: int = AnnFactor.daily
) -> pd.DataFrame:
    if span < 10:
        warnings.warn("it is recommended to use a higher span, e.g 30 days")
    pri_return = to_pri_return(prices=prices)
    assets = prices.columns
    num_assets = len(assets)
    S = np.zeros((num_assets, num_assets))

    for i in range(num_assets):
        for j in range(i, num_assets):
            S[i, j] = S[j, i] = (
                pri_return.iloc[:, [i, j]].ewm(span=180).cov().iloc[-1].iloc[0]
            )
    return pd.DataFrame(S, columns=assets, index=assets) * ann_factor


def to_exponential_correlation_matrix(
    prices: pd.DataFrame, span: int = 180
) -> pd.DataFrame:
    if span < 10:
        warnings.warn("it is recommended to use a higher span, e.g 30 days")
    pri_return = to_pri_return(prices=prices)
    assets = prices.columns
    num_assets = len(assets)
    S = np.zeros((num_assets, num_assets))

    for i in range(num_assets):
        for j in range(i, num_assets):
            S[i, j] = S[j, i] = (
                pri_return.iloc[:, [i, j]].ewm(span=span).corr().iloc[-1].iloc[0]
            )
    return pd.DataFrame(S, columns=assets, index=assets)


@overload
def to_1m(prices: pd.DataFrame) -> pd.Series:
    ...


@overload
def to_1m(prices: pd.Series) -> float:
    ...


def to_1m(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, float]:
    return to_momentum(prices=prices, months=1)


@overload
def to_2m(prices: pd.DataFrame) -> pd.Series:
    ...


@overload
def to_2m(prices: pd.Series) -> float:
    ...


def to_2m(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, float]:
    return to_momentum(prices=prices, months=2)


@overload
def to_3m(prices: pd.DataFrame) -> pd.Series:
    ...


@overload
def to_3m(prices: pd.Series) -> float:
    ...


def to_3m(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, float]:
    return to_momentum(prices=prices, months=3)


@overload
def to_6m(prices: pd.DataFrame) -> pd.Series:
    ...


@overload
def to_6m(prices: pd.Series) -> float:
    ...


def to_6m(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, float]:
    return to_momentum(prices=prices, months=6)


@overload
def to_1y(prices: pd.DataFrame) -> pd.Series:
    ...


@overload
def to_1y(prices: pd.Series) -> float:
    ...


def to_1y(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, float]:
    return to_momentum(prices=prices, months=12)


@overload
def to_calmar_ratio(prices: pd.Series) -> float:
    ...


@overload
def to_calmar_ratio(prices: pd.DataFrame) -> pd.Series:
    ...


def to_calmar_ratio(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, float]:
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_calmar_ratio)

    return to_ann_return(prices=prices) / abs(to_max_drawdown(prices=prices))


@overload
def to_expost_tracking_error(
    prices: pd.DataFrame,
    prices_bm: pd.Series,
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> pd.Series:
    ...


@overload
def to_expost_tracking_error(
    prices: pd.Series,
    prices_bm: pd.Series,
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> float:
    ...


def to_expost_tracking_error(
    prices: Union[pd.DataFrame, pd.Series],
    prices_bm: pd.Series,
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> Union[pd.Series, float]:
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(
            to_expost_tracking_error, prices_bm=prices_bm, ann_factor=ann_factor
        )
    prices_bm = prices_bm.reindex(prices.dropna().index).ffill()
    excess_return = to_pri_return(prices=prices) - to_pri_return(prices=prices_bm)
    excess_return = excess_return.dropna()
    return float(np.std(excess_return)) * (ann_factor**0.5)


def to_exante_tracking_error(
    weights: pd.Series,
    weights_bm: pd.Series,
    prices: Optional[pd.DataFrame] = None,
    covariance_matrix: Optional[pd.DataFrame] = None,
) -> float:
    if covariance_matrix is None:
        if prices is None:
            raise ValueError("You must pass in one of prices or covariance matrix.")
        covariance_matrix = to_covariance_matrix(prices=prices)
    active_weights = weights.subtract(weights_bm, fill_value=0)
    tracking_variance = np.dot(
        np.dot(active_weights, covariance_matrix), active_weights
    )
    tracking_error = np.sqrt(tracking_variance)
    return tracking_error


@overload
def to_information_ratio(
    prices: pd.DataFrame,
    prices_bm: pd.Series,
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> pd.Series:
    ...


@overload
def to_information_ratio(
    prices: pd.Series,
    prices_bm: pd.Series,
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> float:
    ...


def to_information_ratio(
    prices: Union[pd.DataFrame, pd.Series],
    prices_bm: pd.Series,
    ann_factor: Union[int, float] = AnnFactor.daily,
) -> Union[pd.Series, float]:
    """
    Calculates the Information Ratio.

    The Information Ratio measures the excess return of a strategy per unit of tracking error
    relative to a benchmark.

    Args:
        returns (np.ndarray): Array of strategy returns.
        benchmark_returns (np.ndarray): Array of benchmark returns.

    Returns:
        float: The calculated Information Ratio.
    """
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(
            to_information_ratio, prices_bm=prices_bm, ann_factor=ann_factor
        )

    excess_return = to_pri_return(prices=prices) - to_pri_return(prices=prices_bm)
    excess_prices = excess_return.add(1).cumprod()
    return to_ann_return(excess_prices) / to_ann_volatility(excess_prices)


@overload
def to_hurst_exponent(prices: pd.DataFrame, lags: int = 20) -> pd.Series:
    ...


@overload
def to_hurst_exponent(prices: pd.Series, lags: int = 20) -> float:
    ...


def to_hurst_exponent(
    prices: Union[pd.DataFrame, pd.Series], lags: int = 20
) -> Union[pd.Series, float]:
    """
    Calculates the Hurst exponent for a given set of prices.

    The Hurst exponent is a measure of the long-term memory of a time series. It quantifies
    the degree of persistence or trendiness in the data. A value between 0.5 and 1 indicates
    positive autocorrelation and a trend that is likely to persist, while a value between 0 and
    0.5 indicates negative autocorrelation and a trend that is likely to reverse.

    Args:
        prices (Union[pd.DataFrame, pd.Series]): The price data used to calculate the Hurst exponent.
            It can be either a DataFrame with multiple price series or a single Series.
        lags (Optional[int], optional): The number of lags used to calculate the Hurst exponent.
            If not specified, the maximum number of lags possible will be used. Default is None.

    Returns:
        Union[pd.Series, float]: The Hurst exponent as a single value or a Series of Hurst exponents
        if multiple price series are provided.

    References:
        - Hurst, H. E. (1951). Long-term storage capacity of reservoirs.
          Transactions of the American Society of Civil Engineers, 116, 770-799.
        - Lo, A. W. (1991). Long-term memory in stock market prices.
          Econometrica: Journal of the Econometric Society, 1279-1313.

    """
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_hurst_exponent, lags=lags)

    ts = prices.dropna().values

    # variances of the lagged differences
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in range(2, lags)]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(range(2, lags)), np.log(tau), 1)

    return reg[0]


def to_beta(
    prices: Union[pd.DataFrame, pd.Series], prices_bm: pd.Series
) -> Union[pd.Series, float]:
    if isinstance(prices, pd.DataFrame):
        return prices.aggregate(to_beta, prices_bm=prices_bm)
    model = LinearRegression()

    itx = prices.dropna().index.intersection(prices_bm.dropna().index)
    model.fit(
        X=to_pri_return(prices=prices.loc[itx].to_frame()),
        y=to_pri_return(prices=prices_bm.loc[itx]),
    )
    return model.coef_[0]


def to_treynor_ratio(
    prices: Union[pd.DataFrame, pd.Series],
    prices_bm: pd.Series,
    risk_free: Union[int, float] = 0.0,
) -> Union[pd.Series, float]:
    return (to_ann_return(prices=prices) - risk_free) / to_beta(
        prices=prices, prices_bm=prices_bm
    )


def to_jensens_alpha(
    prices: Union[pd.DataFrame, pd.Series],
    prices_bm: pd.Series,
    risk_free: Union[int, float] = 0.0,
) -> Union[pd.Series, float]:
    return to_ann_return(prices=prices) / (
        risk_free
        - to_beta(prices=prices, prices_bm=prices_bm)
        * (to_ann_return(prices=prices_bm) - risk_free)
    )
