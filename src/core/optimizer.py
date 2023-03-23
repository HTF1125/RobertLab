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

    @staticmethod
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


class Optimizer:
    """portfolio optimizer"""

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
        
        if expected_returns is not None:
            self.expected_returns = expected_returns
            self.assets = self.expected_returns.index
        
        if covariance_matrix is not None:
            self.covariance_matrix = covariance_matrix
            
        
        
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.prices_assets = prices_assets
        self.risk_free = risk_free
        self.prices_bm = prices_bm
        self.weights_bm = weights_bm
        self.num_assets = len(self.covariance_matrix)
        self.constraints: Dict = {}
        self.metrics: OptMetrics = OptMetrics()

        






    def set_min_weight(self, min_weight: float = 0.0) -> None:
        """set minimum weights constraint"""
        self.constraints.update(
            {
                "min_weight": {
                    "type": "ineq",
                    "fun": lambda w: w - min_weight,
                }
            }
        )

    def set_max_weight(self, max_weight: float = 1.0) -> None:
        """set maximum weights constraint"""
        self.constraints.update(
            {
                "max_weight": {
                    "type": "ineq",
                    "fun": lambda w: max_weight - w,
                }
            }
        )

    def set_sum_weight(self, sum_weight: float = 1.0) -> None:
        """set summation weights constriant"""
        self.constraints.update(
            {
                "sum_weight": {
                    "type": "eq",
                    "fun": lambda w: np.sum(w) - sum_weight,
                }
            }
        )

    def set_min_return(self, min_return: float = 0.05) -> None:
        """set minimum return constraint"""
        if self.expected_returns is None:
            warnings.warn("unable to set minimum return constraint.")
            warnings.warn("expected returns is null.")
        self.constraints.update(
            {
                "min_return": {
                    "type": "ineq",
                    "fun": lambda w: self.metrics.expected_return(
                        weights=w, expected_returns=self.expected_returns.values
                    )
                    - min_return,
                }
            }
        )

    def set_max_return(self, max_return: float = 0.05) -> None:
        """set maximum return constraint"""
        if self.expected_returns is None:
            warnings.warn("unable to set maximum return constraint.")
            warnings.warn("expected returns is null.")
        self.constraints.update(
            {
                "max_return": {
                    "type": "ineq",
                    "fun": lambda w: max_return
                    - self.metrics.expected_return(
                        weights=w, expected_returns=self.expected_returns.values
                    ),
                }
            }
        )

    def set_min_volatility(self, min_volatility: float = 0.05) -> None:
        """set minimum volatility constraint"""
        if self.expected_returns is None:
            warnings.warn("unable to set minimum volatility constraint.")
            warnings.warn("expected returns is null.")
        self.constraints.update(
            {
                "min_volatility": {
                    "type": "ineq",
                    "fun": lambda w: self.metrics.expected_volatility(
                        weights=w, covariance_matrix=self.covariance_matrix.values
                    )
                    - min_volatility,
                }
            }
        )

    def set_max_volatility(self, max_volatility: float = 0.05) -> None:
        """set maximum volatility constraint"""
        if self.expected_returns is None:
            warnings.warn("unable to set maximum volatility constraint.")
            warnings.warn("expected returns is null.")
        self.constraints.update(
            {
                "max_volatility": {
                    "type": "ineq",
                    "fun": lambda w: max_volatility
                    - self.metrics.expected_volatility(
                        weights=w, covariance_matrix=self.covariance_matrix.values
                    ),
                }
            }
        )

    def set_max_active_weight(self, max_active_weight: float = 0.10) -> None:
        """set maximum active weight against benchmark"""
        if self.weights_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
        self.constraints.update(
            {
                "max_active_weight": {
                    "type": "ineq",
                    "fun": lambda w: max_active_weight
                    - np.sum(np.abs(w - self.weights_bm.values)),
                }
            }
        )

    def set_max_exante_tracking_error(
        self, max_exante_tracking_error: float = 0.02
    ) -> None:
        """set maximum exante tracking error constraint"""

        self.constraints.update(
            {
                "max_exante_tracking_error": {
                    "type": "ineq",
                    "fun": lambda w: max_exante_tracking_error
                    - self.metrics.exante_tracking_error(
                        weights=w,
                        weights_bm=self.weights_bm.values,
                        covariance_matrix=self.covariance_matrix.values,
                    ),
                }
            }
        )

    def set_max_expost_tracking_error(
        self, max_expost_tracking_error: float = 0.02
    ) -> None:
        """set maximum expost tracking error constraint"""

        itx = self.prices_assets.dropna().index.intersection(
            self.prices_bm.dropna().index
        )

        pri_returns_assets = self.prices_assets.loc[itx].pct_change().fillna(0)
        pri_returns_bm = self.prices_bm.loc[itx].pct_change().fillna(0)

        self.constraints.update(
            {
                "max_expost_tracking_error": {
                    "type": "ineq",
                    "fun": lambda w: max_expost_tracking_error
                    - self.metrics.expost_tracking_error(
                        weights=w,
                        pri_returns_assets=pri_returns_assets.values,
                        pri_returns_bm=pri_returns_bm.values,
                    ),
                }
            }
        )


