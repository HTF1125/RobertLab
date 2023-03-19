from typing import Optional
from datetime import datetime
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
    return prices.pct_change().fillna(0)


def to_log_return(prices: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    return to_pri_returns(prices=prices).apply(np.log1p)


def get_startdate(prices: pd.DataFrame) -> datetime:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        datetime: _description_
    """
    return parser.parse(str(prices.index[0]))


def get_enddate(prices: pd.DataFrame) -> datetime:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        datetime: _description_
    """
    return parser.parse(str(prices.index[-1]))


def to_num_year(prices: pd.DataFrame) -> float:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        float: _description_
    """
    start = get_startdate(prices=prices)
    end = get_enddate(prices=prices)
    return (end - start).days / 365.0


def to_ann_factor(prices: pd.DataFrame) -> float:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return len(prices) / to_num_year(prices=prices)


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
        to_pri_returns(prices=prices).add(1).prod() ** (1 / to_num_year(prices=prices))
        - 1
    )


def to_ann_variances(prices: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return to_pri_returns(prices=prices).var() * to_ann_factor(prices=prices)


def to_ann_volatilites(prices: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return to_ann_variances(prices=prices).apply(np.sqrt)


def to_ann_semi_variances(
    prices: pd.DataFrame, ann_factor: Optional[float] = None
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    pri_returns = to_pri_returns(prices=prices)
    positive_pri_returns = pri_returns[pri_returns >= 0]
    if not ann_factor:
        ann_factor = to_ann_factor(prices=prices)
    return positive_pri_returns.var() * ann_factor


def to_ann_semi_volatilities(
    prices: pd.DataFrame, ann_factor: Optional[float] = None
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        ann_factors (Optional[float], optional): _description_. Defaults to None.

    Returns:
        pd.Series: _description_
    """
    return to_ann_semi_variances(prices=prices, ann_factor=ann_factor) ** 0.5


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


