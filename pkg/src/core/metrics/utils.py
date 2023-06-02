from typing import Tuple, List
import numpy as np
import pandas as pd


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
    if len(sorted_tree) < 2:
        return

    left = sorted_tree[0 : int(len(sorted_tree) / 2)]
    right = sorted_tree[int(len(sorted_tree) / 2) :]

    if len(left) > 2 and len(right) > 2:
        return [(left, right), recursive_bisection(left), recursive_bisection(right)]
    return (left, right)


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
