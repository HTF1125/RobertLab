from typing import Optional, List
import numpy as np


def expected_return(
    weights: np.ndarray,
    expected_returns: np.ndarray,
) -> float:
    """
    Portfolio expected return.

    Args:
        weight (np.ndarray): weight of assets.
        expected_returns (np.ndarray): expected return of assets.

    Returns:
        float: portfolio expected return.
    """
    return np.dot(weights, expected_returns)


def expected_variance(
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
) -> float:
    """
    Portfolio expected variance.

    Args:
        weight (np.ndarray): weight of assets.
        covariance_matrix (np.ndarray): covariance matrix of assets.

    Returns:
        float: portfolio expected variance.
    """
    return np.linalg.multi_dot((weights, covariance_matrix, weights))


def expected_volatility(
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
    sub_covariance_matrix_idx: Optional[List] = None,
) -> float:
    """risk contributions"""

    if sub_covariance_matrix_idx:
        sub_covariance_matrix = covariance_matrix.copy()
        for i, row in enumerate(covariance_matrix):
            for j, _ in enumerate(row):
                if (
                    i not in sub_covariance_matrix_idx
                    and j not in sub_covariance_matrix_idx
                ):
                    sub_covariance_matrix[i, j] = 0
        return np.sqrt(
            expected_variance(weights=weights, covariance_matrix=sub_covariance_matrix)
        )

    return np.sqrt(
        expected_variance(weights=weights, covariance_matrix=covariance_matrix)
    )


def expected_sharpe(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free: float = 0.0,
) -> float:
    """
    Portfolio expected sharpe ratio.

    Args:
        weight (np.ndarray): weight of assets.
        expected_returns (np.ndarray): expected return of assets.
        covariance_matrix (np.ndarray): covariance matrix of assets.

    Returns:
        float: portfolio expected sharpe ratio.
    """
    ret = expected_return(weights=weights, expected_returns=expected_returns)
    std = expected_volatility(weights=weights, covariance_matrix=covariance_matrix)
    return (ret - risk_free) / std


def l1_norm(vals: np.ndarray, gamma: float = 1) -> float:
    """_summary_

    Args:
        vals (np.ndarray): _description_
        gamma (float, optional): _description_. Defaults to 1.

    Returns:
        float: _description_
    """
    return np.abs(vals).sum() * gamma


def l2_norm(vals: np.ndarray, gamma: float = 1) -> float:
    """
    L2 regularization.

    Args:
        weight (np.ndarray): asset weight in the portfolio.
        gamma (float, optional): L2 regularisation parameter. Defaults to 1.
            Increase if you want more non-negligible weight.

    Returns:
        float: L2 regularization.
    """
    return np.sum(np.square(vals)) * gamma


def exante_tracking_error(
    weights: np.ndarray, weights_bm: np.ndarray, covariance_matrix: np.ndarray
) -> float:
    """
    Calculate the ex-ante tracking error.

    Maths:
        formula here.

    Args:
        weight (np.ndarray): asset weight in the portfolio.
        weight_benchmark (np.ndarray): benchmarket weight of the portfolio.
        covaraince_matrix (np.ndarray): asset covariance matrix.

    Returns:
        float: ex-ante tracking error.
    """
    rel_weight = np.subtract(weights, weights_bm)
    tracking_variance = np.dot(np.dot(rel_weight, covariance_matrix), rel_weight)
    tracking_error = np.sqrt(tracking_variance)
    return tracking_error


def expost_tracking_error(
    weights: np.ndarray,
    pri_returns_assets: np.ndarray,
    pri_returns_bm: np.ndarray,
) -> float:
    """_summary_

    Args:
        weights (np.ndarray): _description_
        pri_returns_assets (np.ndarray): _description_
        pri_returns_benchmark (np.ndarray): _description_

    Returns:
        float: _description_
    """
    rel_return = np.dot(pri_returns_assets, weights) - pri_returns_bm
    mean = np.sum(rel_return) / len(rel_return)
    return np.sum(np.square(rel_return - mean))


def risk_contributions(
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
    sub_covariance_matrix_idx: Optional[List] = None,
) -> np.ndarray:
    """risk contributions"""
    volatility = expected_volatility(
        weights=weights, covariance_matrix=covariance_matrix
    )
    if sub_covariance_matrix_idx:
        sub_covariance_matrix = covariance_matrix.copy()
        for i, row in enumerate(covariance_matrix):
            for j, _ in enumerate(row):
                if (
                    i not in sub_covariance_matrix_idx
                    and j not in sub_covariance_matrix_idx
                ):
                    sub_covariance_matrix[i, j] = 0
        return np.dot(sub_covariance_matrix, weights) * weights / volatility

    return np.dot(covariance_matrix, weights) * weights / volatility
