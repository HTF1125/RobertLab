from typing import Optional
from dateutil import parser
import numpy as np
import pandas as pd


def to_pri_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    def to_pri_return(price: pd.Series) -> float:
        return price.dropna().pct_change().fillna(0)

    return prices.apply(to_pri_return)


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    return to_pri_returns(prices=prices).apply(np.log1p)


def to_num_years(prices: pd.DataFrame) -> float:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        float: _description_
    """

    def to_num_year(price) -> float:
        p = price.dropna()
        start = parser.parse(str(p.index[0]))
        end = parser.parse(str(p.index[-1]))
        return (end - start).days / 365.0

    return prices.apply(to_num_year, axis=0)


def to_num_bars(prices: pd.DataFrame) -> float:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        float: _description_
    """

    def to_num_bar(price) -> float:
        return len(price.dropna())

    return prices.apply(to_num_bar, axis=0)


def to_ann_factors(prices: pd.DataFrame) -> float:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return to_num_bars(prices=prices) / to_num_years(prices=prices)


def to_cum_returns(prices: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return to_pri_returns(prices=prices).add(1).prod() - 1


def to_ann_returns(prices: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return (
        to_pri_returns(prices=prices).add(1).prod() ** (1 / to_num_years(prices=prices))
        - 1
    )


def to_ann_variances(
    prices: pd.DataFrame, ann_factors: Optional[float] = None
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    if not ann_factors:
        ann_factors = to_ann_factors(prices=prices)
    return to_pri_returns(prices=prices).var() * to_ann_factors(prices=prices)


def to_ann_volatilites(
    prices: pd.DataFrame, ann_factors: Optional[float] = None
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return to_ann_variances(prices=prices, ann_factors=ann_factors).apply(np.sqrt)


def to_ann_semi_variances(
    prices: pd.DataFrame, ann_factors: Optional[float] = None
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    pri_returns = to_pri_returns(prices=prices)
    positive_pri_returns = pri_returns[pri_returns >= 0]
    if not ann_factors:
        ann_factors = to_ann_factors(prices=prices)
    return positive_pri_returns.var() * ann_factors


def to_ann_semi_volatilities(
    prices: pd.DataFrame, ann_factors: Optional[float] = None
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        ann_factors (Optional[float], optional): _description_. Defaults to None.

    Returns:
        pd.Series: _description_
    """
    return to_ann_semi_variances(prices=prices, ann_factors=ann_factors) ** 0.5


def to_drawdown(
    prices: pd.DataFrame,
    window: Optional[int] = None,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        window (Optional[int], optional): _description_. Defaults to None.

    Returns:
        pd.Series: _description_
    """
    if window:
        return prices / prices.rolling(window=window, min_periods=min_periods).max() - 1
    return prices / prices.expanding(min_periods=min_periods or 1).max() - 1


def to_max_drawdown(
    prices: pd.DataFrame,
    window: Optional[int] = None,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        window (Optional[int], optional): _description_. Defaults to None.
        min_periods (Optional[int], optional): _description_. Defaults to None.

    Returns:
        pd.Series: _description_
    """
    return to_drawdown(prices=prices, window=window, min_periods=min_periods).min()


def to_sharpe_ratios(
    prices: pd.DataFrame, risk_free: float = 0.0, ann_factors: Optional[float] = None
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        risk_free (float, optional): _description_. Defaults to 0..
        ann_factors (Optional[float], optional): _description_. Defaults to None.

    Returns:
        pd.Series: _description_
    """
    excess_returns = to_ann_returns(prices=prices) - risk_free
    return excess_returns / to_ann_volatilites(prices=prices, ann_factors=ann_factors)


def to_sortino_ratios(
    prices: pd.DataFrame, ann_factors: Optional[float] = None
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        ann_factors (Optional[float], optional): _description_. Defaults to None.

    Returns:
        pd.Series: _description_
    """
    if not ann_factors:
        ann_factors = to_ann_factors(prices=prices)

    ann_returns = to_ann_returns(prices=prices)
    ann_semi_volatilities = to_ann_semi_volatilities(
        prices=prices, ann_factors=ann_factors
    )

    return ann_returns / ann_semi_volatilities

