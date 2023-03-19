from datetime import datetime
from dateutil import parser
import numpy as np
import pandas as pd


def to_pri_return(prices: pd.DataFrame) -> pd.DataFrame:
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
    return to_pri_return(prices=prices).apply(np.log1p)


def get_startdate(prices: pd.DataFrame) -> datetime:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        datetime: _description_
    """
    return parser.parse(prices.index[0])


def get_enddate(prices: pd.DataFrame) -> datetime:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        datetime: _description_
    """
    return parser.parse(prices.index[0])


def n_years(prices: pd.DataFrame) -> float:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        float: _description_
    """
    start = get_startdate(prices=prices)
    end = get_enddate(prices=prices)
    return (end - start).days / 365.0


def ann_factors(prices: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return len(prices) / n_years(prices=prices)


def cum_returns(prices: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return to_pri_return(prices=prices).add(1).prod() - 1


def ann_returns(prices: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return (
        to_pri_return(prices=prices).add(1).prod() ** (1 / n_years(prices=prices)) - 1
    )


def ann_variances(prices: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return to_pri_return(prices=prices).var() * ann_factors(prices=prices)


def ann_volatilites(prices: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return ann_variances(prices=prices).apply(np.sqrt)
