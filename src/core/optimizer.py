import warnings
from typing import Optional, Callable, Dict, List
from functools import partial
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
from src.core.metrics import cov_to_corr, recursive_bisection


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
        return np.linalg.multi_dot((weights, covariance_matrix, weights))

    def expected_volatility(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        sub_covariance_matrix_idx: Optional[List] = None,
    ) -> float:
        """risk contributions"""

        if sub_covariance_matrix_idx:
            sub_covariance_matrix = covariance_matrix.copy()
            for i, row in enumerate(covariance_matrix):
                for j, val in enumerate(row):
                    if (
                        i not in sub_covariance_matrix_idx
                        and j not in sub_covariance_matrix_idx
                    ):
                        sub_covariance_matrix[i, j] = 0
            return np.sqrt(
                self.expected_variance(
                    weights=weights, covariance_matrix=sub_covariance_matrix
                )
            )

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
    def l1_norm(vals: np.ndarray, gamma: float = 1) -> float:
        """_summary_

        Args:
            vals (np.ndarray): _description_
            gamma (float, optional): _description_. Defaults to 1.

        Returns:
            float: _description_
        """
        return np.abs(vals).sum() * gamma

    @staticmethod
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

    def risk_contributions(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        sub_covariance_matrix_idx: Optional[List] = None,
    ) -> np.ndarray:
        """risk contributions"""
        volatility = self.expected_volatility(
            weights=weights, covariance_matrix=covariance_matrix
        )
        if sub_covariance_matrix_idx:
            sub_covariance_matrix = covariance_matrix.copy()
            for i, row in enumerate(covariance_matrix):
                for j, val in enumerate(row):
                    if (
                        i not in sub_covariance_matrix_idx
                        and j not in sub_covariance_matrix_idx
                    ):
                        sub_covariance_matrix[i, j] = 0
            return np.dot(sub_covariance_matrix, weights) * weights / volatility

        return np.dot(covariance_matrix, weights) * weights / volatility


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
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        sum_weight: float = 1.0,
        min_return: Optional[float] = None,
        max_return: Optional[float] = None,
        min_volatility: Optional[float] = None,
        max_volatility: Optional[float] = None,
        active_weight: Optional[float] = None,
        exante_tracking_error: Optional[float] = None,
        expost_tracking_error: Optional[float] = None,
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
        self.constraints: List = []
        self.metrics: OptMetrics = OptMetrics()

        self.set_min_weight(min_weight=min_weight)
        self.set_max_weight(max_weight=max_weight)
        self.set_sum_weight(sum_weight=sum_weight)

        if min_return:
            self.set_min_return(min_return=min_return)

        if max_return:
            self.set_max_return(max_return=max_return)

        if min_volatility:
            self.set_min_volatility(min_volatility=min_volatility)

        if max_volatility:
            self.set_max_volatility(max_volatility=max_volatility)

        if active_weight:
            self.set_max_active_weight(active_weight=active_weight)

        if exante_tracking_error:
            self.set_max_exante_tracking_error(
                exante_tracking_error=exante_tracking_error
            )

        if expost_tracking_error:
            self.set_max_expost_tracking_error(
                expost_tracking_error=expost_tracking_error
            )

    @property
    def expected_returns(self) -> pd.Series:
        """expected_returns"""
        return self._expected_returns

    @expected_returns.setter
    def expected_returns(self, expected_returns: Optional[pd.Series] = None) -> None:
        self._expected_returns = expected_returns
        if expected_returns is not None:
            self.assets = self.expected_returns.index

    @property
    def covariance_matrix(self) -> pd.DataFrame:
        """covariance_matrix"""
        return self._covariance_matrix

    @covariance_matrix.setter
    def covariance_matrix(
        self, covariance_matrix: Optional[pd.DataFrame] = None
    ) -> None:
        self._covariance_matrix = covariance_matrix
        if covariance_matrix is not None:
            self.assets = self.covariance_matrix.index
            self.assets = self.covariance_matrix.columns

    @property
    def prices_assets(self) -> pd.DataFrame:
        """prices_assets"""
        return self._prices_assets

    @prices_assets.setter
    def prices_assets(self, prices_assets: Optional[pd.DataFrame] = None) -> None:
        self._prices_assets = prices_assets
        if prices_assets is not None:
            self.assets = self.prices_assets.columns

    @property
    def assets(self) -> pd.Index:
        """assets"""
        try:
            return self._assets
        except AttributeError:
            return None

    @assets.setter
    def assets(self, assets: pd.Index) -> None:
        """assets setter"""

        if self.assets is not None:
            assert self.assets.equals(assets)
            return
        self._assets = assets

    @property
    def num_asset(self) -> int:
        """return number of asset"""
        return len(self.assets)

    def set_min_weight(self, min_weight: float = 0.0) -> None:
        """set minimum weights constraint"""
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: w - min_weight,
            }
        )

    def set_max_weight(self, max_weight: float = 1.0) -> None:
        """set maximum weights constraint"""
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: max_weight - w,
            }
        )

    def set_sum_weight(self, sum_weight: float = 1.0) -> None:
        """set summation weights constriant"""
        self.constraints.append(
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - sum_weight,
            }
        )

    def set_min_return(self, min_return: float = 0.05) -> None:
        """set minimum return constraint"""
        if self.expected_returns is None:
            warnings.warn("unable to set minimum return constraint.")
            warnings.warn("expected returns is null.")
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: self.metrics.expected_return(
                    weights=w, expected_returns=self.expected_returns.values
                )
                - min_return,
            }
        )

    def set_max_return(self, max_return: float = 0.05) -> None:
        """set maximum return constraint"""
        if self.expected_returns is None:
            warnings.warn("unable to set maximum return constraint.")
            warnings.warn("expected returns is null.")
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: max_return
                - self.metrics.expected_return(
                    weights=w, expected_returns=self.expected_returns.values
                ),
            }
        )

    def set_min_volatility(self, min_volatility: float = 0.05) -> None:
        """set minimum volatility constraint"""
        if self.expected_returns is None:
            warnings.warn("unable to set minimum volatility constraint.")
            warnings.warn("expected returns is null.")
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: self.metrics.expected_volatility(
                    weights=w, covariance_matrix=self.covariance_matrix.values
                )
                - min_volatility,
            }
        )

    def set_max_volatility(self, max_volatility: float = 0.05) -> None:
        """set maximum volatility constraint"""
        if self.expected_returns is None:
            warnings.warn("unable to set maximum volatility constraint.")
            warnings.warn("expected returns is null.")
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: max_volatility
                - self.metrics.expected_volatility(
                    weights=w, covariance_matrix=self.covariance_matrix.values
                ),
            }
        )

    def set_max_active_weight(self, active_weight: float = 0.10) -> None:
        """set maximum active weight against benchmark"""
        if self.weights_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: active_weight
                - np.sum(np.abs(w - self.weights_bm.values)),
            }
        )

    def set_max_exante_tracking_error(
        self, exante_tracking_error: float = 0.02
    ) -> None:
        """set maximum exante tracking error constraint"""

        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: exante_tracking_error
                - self.metrics.exante_tracking_error(
                    weights=w,
                    weights_bm=self.weights_bm.values,
                    covariance_matrix=self.covariance_matrix.values,
                ),
            }
        )

    def set_max_expost_tracking_error(
        self, expost_tracking_error: float = 0.02
    ) -> None:
        """set maximum expost tracking error constraint"""

        itx = self.prices_assets.dropna().index.intersection(
            self.prices_bm.dropna().index
        )

        pri_returns_assets = self.prices_assets.loc[itx].pct_change().fillna(0)
        pri_returns_bm = self.prices_bm.loc[itx].pct_change().fillna(0)

        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: expost_tracking_error
                - self.metrics.expost_tracking_error(
                    weights=w,
                    pri_returns_assets=pri_returns_assets.values,
                    pri_returns_bm=pri_returns_bm.values,
                ),
            }
        )

    def solve(
        self, objective: Callable, extra_constraints: Optional[List[Dict]] = None
    ) -> Optional[pd.Series]:
        constraints = self.constraints.copy()
        if extra_constraints:
            constraints.extend(extra_constraints)
        problem = minimize(
            fun=objective,
            method="SLSQP",
            constraints=constraints,
            x0=np.ones(shape=self.num_asset) / self.num_asset,
        )

        if problem.success:
            return pd.Series(data=problem.x, index=self.assets, name="weights").round(6)
        return None

    def maximized_return(self) -> Optional[pd.Series]:
        """calculate max return weights"""
        return self.solve(
            objective=partial(
                self.metrics.expected_return,
                expected_returns=self.expected_returns.values * -1,
            )
        )

    def minimized_volatility(self) -> Optional[pd.Series]:
        """_summary_

        Returns:
            Optional[pd.Series]: _description_
        """
        return self.solve(
            objective=partial(
                self.metrics.expected_volatility,
                covariance_matrix=self.covariance_matrix.values,
            )
        )

    def maximized_sharpe_ratio(self) -> Optional[pd.Series]:
        """_summary_

        Returns:
            Optional[pd.Series]: _description_
        """
        return self.solve(
            objective=partial(
                self.metrics.expected_sharpe,
                expected_returns=self.expected_returns.values,
                covariance_matrix=self.covariance_matrix.values,
                risk_free=self.risk_free,
            )
        )

    def hierarchical_equal_risk_contribution(self) -> Optional[pd.Series]:
        """calculate herc weights"""
        corr = cov_to_corr(self.covariance_matrix.values)
        dist = np.sqrt((1 - corr).round(5) / 2)
        clusters = linkage(squareform(dist), method="single")
        sorted_tree = list(to_tree(clusters, rd=False).pre_order())
        cluster_sets = recursive_bisection(sorted_tree)
        return self.solve(
            objective=lambda w: self.metrics.l2_norm(
                np.array(
                    [
                        np.sum(
                            self.metrics.risk_contributions(
                                weights=w,
                                covariance_matrix=self.covariance_matrix.values,
                                sub_covariance_matrix_idx=left_idx,
                            )
                        )
                        - np.sum(
                            self.metrics.risk_contributions(
                                weights=w,
                                covariance_matrix=self.covariance_matrix.values,
                                sub_covariance_matrix_idx=right_idx,
                            )
                        )
                        for left_idx, right_idx in cluster_sets
                    ]
                )
            )
        )

    def hierarchical_risk_parity(self) -> Optional[pd.Series]:
        """calculate herc weights"""

        corr = cov_to_corr(self.covariance_matrix.values)
        dist = np.sqrt((1 - corr).round(5) / 2)
        clusters = linkage(squareform(dist), method="single")
        sorted_tree = list(to_tree(clusters, rd=False).pre_order())
        cluster_sets = recursive_bisection(sorted_tree)
        return self.solve(
            objective=lambda w: self.metrics.l2_norm(
                np.array(
                    [
                        self.metrics.expected_volatility(
                            weights=w,
                            covariance_matrix=self.covariance_matrix.values,
                            sub_covariance_matrix_idx=left_idx,
                        )
                        - self.metrics.expected_volatility(
                            weights=w,
                            covariance_matrix=self.covariance_matrix.values,
                            sub_covariance_matrix_idx=right_idx,
                        )
                        for left_idx, right_idx in cluster_sets
                    ]
                )
            )
        )

    def risk_parity(self, budgets: Optional[np.ndarray] = None) -> Optional[pd.Series]:
        """_summary_

        Returns:
            Optional[pd.Series]: _description_
        """
        if budgets is None:
            budgets = np.ones(self.num_asset) / self.num_asset
        return self.solve(
            objective=lambda w: self.metrics.l2_norm(
                np.subtract(
                    self.metrics.risk_contributions(
                        weights=w, covariance_matrix=self.covariance_matrix.values
                    ),
                    np.multiply(
                        budgets,
                        self.metrics.expected_volatility(
                            weights=w, covariance_matrix=self.covariance_matrix.values
                        ),
                    ),
                )
            )
        )

    def inverse_variance(self) -> Optional[pd.Series]:
        """_summary_

        Returns:
            Optional[pd.Series]: _description_
        """
        inv_var_weights = 1 / np.diag(self.covariance_matrix.values)
        inv_var_weights /= inv_var_weights.sum()
        return self.solve(
            objective=lambda w: self.metrics.l2_norm(np.subtract(w, inv_var_weights))
        )
