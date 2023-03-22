import warnings
from typing import Optional, Dict
from scipy.optimize import minimize
import numpy as np
import pandas as pd


class OptMetrics:
    """portfolio optimizer metrics"""

    @staticmethod
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

    @staticmethod
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
        return np.dot(np.dot(weights, covariance_matrix), weights)

    def expected_volatility(
        self, weights: np.ndarray, covariance_matrix: np.ndarray
    ) -> float:
        """
        Portfolio expected volatility.

        Args:
            weight (np.ndarray): weight of assets.
            covariance_matrix (np.ndarray): covariance matrix of assets.

        Returns:
            float: portfolio expected volatility.
        """
        return np.sqrt(
            self.expected_variance(weights=weights, covariance_matrix=covariance_matrix)
        )

    def expected_sharpe(
        self,
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
        ret = self.expected_return(weights=weights, expected_returns=expected_returns)
        std = self.expected_volatility(
            weights=weights, covariance_matrix=covariance_matrix
        )
        return (ret - risk_free) / std

    @staticmethod
    def l1_norm(weights: np.ndarray, gamma: float = 1) -> float:
        """
        L1 regularization.

        Args:
            weight (np.ndarray): asset weight in the portfolio.
            gamma (float, optional): L2 regularisation parameter. Defaults to 1.
                Increase if you want more non-negligible weight.

        Returns:
            float: L2 regularization.
        """
        return np.abs(weights).sum() * gamma

    @staticmethod
    def l2_norm(weights: np.ndarray, gamma: float = 1) -> float:
        """
        L2 regularization.

        Args:
            weight (np.ndarray): asset weight in the portfolio.
            gamma (float, optional): L2 regularisation parameter. Defaults to 1.
                Increase if you want more non-negligible weight.

        Returns:
            float: L2 regularization.
        """
        return np.sum(np.square(weights)) * gamma

    @staticmethod
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


class Optimizer:
    """portfolio optimizer"""

    __constraints__: Dict = {}
    __metrics__: OptMetrics = OptMetrics()

    def __init__(
        self,
        expected_returns: Optional[pd.Series] = None,
        covariance_matrix: Optional[pd.DataFrame] = None,
        risk_free: float = 0.0,
        prices_assets: Optional[pd.DataFrame] = None,
        prices_bm: Optional[pd.Series] = None,
        weights_bm: Optional[pd.Series] = None,
    ) -> None:
        """initialization"""
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.prices_assets = prices_assets
        self.risk_free = risk_free
        self.prices_bm = prices_bm
        self.weights_bm = weights_bm
        self.num_assets = len(self.covariance_matrix)

    def set_min_weights(self, min_weights: float = 0.0) -> None:
        """set minimum weights constraint"""
        self.__constraints__.update(
            {
                "min_weights": {
                    "type": "ineq",
                    "fun": lambda w: w - min_weights,
                }
            }
        )

    def set_max_weights(self, max_weights: float = 1.0) -> None:
        """set maximum weights constraint"""
        self.__constraints__.update(
            {
                "max_weights": {
                    "type": "ineq",
                    "fun": lambda w: max_weights - w,
                }
            }
        )

    def set_sum_weights(self, sum_weights: float = 1.0) -> None:
        """set summation weights constriant"""
        self.__constraints__.update(
            {
                "sum_weights": {
                    "sum_weights": {
                        "type": "eq",
                        "fun": lambda w: np.sum(w) - sum_weights,
                    }
                }
            }
        )
