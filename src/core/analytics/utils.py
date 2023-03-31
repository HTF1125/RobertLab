from typing import Optional, Union, Tuple, List
import numpy as np
import pandas as pd
from .metrics import to_pri_return, to_ann_factor

def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        cov (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    vol = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vol, vol)
    corr[corr < -1], corr[corr > 1] = -1, 1
    return corr


def recursive_bisection(sorted_tree) -> List[Tuple[List[int], List[int]]]:
    """_summary_

    Args:
        sorted_tree (_type_): _description_

    Returns:
        List[Tuple[List[int], List[int]]]: _description_
    """
    if len(sorted_tree) < 3:
        return

    left = sorted_tree[0 : int(len(sorted_tree) / 2)]
    right = sorted_tree[int(len(sorted_tree) / 2) :]

    if len(left) > 2 and len(right) > 2:
        return [(left, right), recursive_bisection(left), recursive_bisection(right)]
    return (left, right)


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
    ann_factors: Optional[Union[int, float, pd.Series]] = None,
    com: Optional[float] = None,
    span: Optional[float] = None,
    halflife: Optional[float] = None,
) -> pd.DataFrame:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        ann_factors (Optional[Union[int, float, pd.Series]], optional): _description_. Defaults to None.
        com (Optional[float], optional): _description_. Defaults to None.
        span (Optional[float], optional): _description_. Defaults to None.
        halflife (Optional[float], optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """

    pri_returns = to_pri_return(prices=prices)
    if ann_factors is None:
        ann_factors = to_ann_factor(prices=prices)

    alpha = exponential_alpha(com=com, span=span, halflife=halflife)

    if alpha is None:
        return pri_returns.cov() * ann_factors

    exp_covariance_matrix = (
        pri_returns.ewm(alpha=alpha).cov().unstack().iloc[-1].unstack() * ann_factors
    )

    return exp_covariance_matrix.loc[prices.columns, prices.columns]

def get_cluster_assets(clusters, node, num_assets) -> List:
    """_summary_

    Args:
        clusters (_type_): _description_
        node (_type_): _description_
        num_assets (_type_): _description_

    Returns:
        List: _description_
    """
    if node < num_assets:
        return [int(node)]
    row = clusters[int(node - num_assets)]
    return get_cluster_assets(clusters, row[0], num_assets) + get_cluster_assets(
        clusters, row[1], num_assets
    )