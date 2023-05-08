from typing import Optional
import numpy as np
import pandas as pd
from .metrics import to_pri_return, to_ann_factor, to_ann_return
from ..ext.periods import AnnFactor


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
    if ann_factor is None:
        ann_factor = to_ann_factor(prices=prices)
    alpha = exponential_alpha(com=com, span=span, halflife=halflife)

    if alpha is None:
        return pri_returns.cov() * ann_factor

    exp_covariance_matrix = (
        pri_returns.ewm(alpha=alpha).cov().unstack().iloc[-1].unstack() * ann_factor
    )

    return exp_covariance_matrix.loc[prices.columns, prices.columns]


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
    alpha = exponential_alpha(com=com, span=span, halflife=halflife)

    if alpha is None:
        return pri_returns.corr()

    exp_covariance_matrix = (
        pri_returns.ewm(alpha=alpha).corr().unstack().iloc[-1].unstack()
    )

    return exp_covariance_matrix.loc[prices.columns, prices.columns]
