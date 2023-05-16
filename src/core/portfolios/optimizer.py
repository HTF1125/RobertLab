import warnings
from typing import Optional, Callable, Dict, List
from functools import partial
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
from . import objectives
from ..analytics import estimators
from ..analytics.utils import cov_to_corr, recursive_bisection


class OptimizerMetrics:
    def __init__(self) -> None:
        self.prices: Optional[pd.DataFrame] = None
        self.expected_returns: Optional[pd.Series] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.assets: Optional[pd.Index] = None


class Optimizer:
    """portfolio optimizer"""

    @classmethod
    def from_prices(cls, prices: pd.DataFrame, **kwargs) -> "Optimizer":
        """_summary_

        Args:
            prices (pd.DataFrame): price of assets.

        Returns:
            Optimizer: initialized optimizer class.
        """
        return cls(
            expected_returns=estimators.to_expected_returns(prices=prices),
            covariance_matrix=estimators.to_covariance_matrix(prices=prices),
            correlation_matrix=estimators.to_correlation_matrix(prices=prices),
            **kwargs,
        )

    def __init__(
        self,
        expected_returns: Optional[pd.Series] = None,
        covariance_matrix: Optional[pd.DataFrame] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        risk_free: float = 0.0,
        prices: Optional[pd.DataFrame] = None,
        prices_bm: Optional[pd.Series] = None,
        weights_bm: Optional[pd.Series] = None,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        sum_weight: float = 1.0,
        min_return: Optional[float] = None,
        max_return: Optional[float] = None,
        min_volatility: Optional[float] = None,
        max_volatility: Optional[float] = None,
        max_active_weight: Optional[float] = None,
        max_exante_tracking_error: Optional[float] = None,
        max_expost_tracking_error: Optional[float] = None,
    ) -> None:
        """initialization"""
        self.metrics = OptimizerMetrics()
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.correlation_matrix = correlation_matrix
        self.prices = prices
        self.risk_free = risk_free
        self.prices_bm = prices_bm
        self.weights_bm = weights_bm
        self.constraints: List = []

        self.set_min_weight(min_weight=min_weight)
        self.set_max_weight(max_weight=max_weight)
        self.set_sum_weight(sum_weight=sum_weight)

        if min_return:
            self.set_min_return(min_return)
        if max_return:
            self.set_max_return(max_return)
        if min_volatility:
            self.set_min_volatility(min_volatility)
        if max_volatility:
            self.set_max_volatility(max_volatility)
        if max_active_weight:
            self.set_max_active_weight(max_active_weight)
        if max_exante_tracking_error:
            self.set_max_exante_tracking_error(max_exante_tracking_error)
        if max_expost_tracking_error:
            self.set_max_expost_tracking_error(max_expost_tracking_error)

    @property
    def expected_returns(self) -> Optional[pd.Series]:
        """expected_returns"""
        return self.metrics.expected_returns

    @expected_returns.setter
    def expected_returns(self, expected_returns: Optional[pd.Series] = None) -> None:
        if expected_returns is not None:
            self.metrics.expected_returns = expected_returns
            self.assets = expected_returns.index

    @property
    def covariance_matrix(self) -> Optional[pd.DataFrame]:
        """covariance_matrix"""
        return self.metrics.covariance_matrix

    @covariance_matrix.setter
    def covariance_matrix(
        self, covariance_matrix: Optional[pd.DataFrame] = None
    ) -> None:
        if covariance_matrix is not None:
            self.metrics.covariance_matrix = covariance_matrix
            self.assets = covariance_matrix.index
            self.assets = covariance_matrix.columns

    @property
    def correlation_matrix(self) -> Optional[pd.DataFrame]:
        """correlation_matrix"""
        return self.metrics.correlation_matrix

    @correlation_matrix.setter
    def correlation_matrix(
        self, correlation_matrix: Optional[pd.DataFrame] = None
    ) -> None:
        if correlation_matrix is not None:
            self.metrics.correlation_matrix = correlation_matrix
            self.assets = correlation_matrix.index
            self.assets = correlation_matrix.columns

    @property
    def prices(self) -> Optional[pd.DataFrame]:
        """prices_assets"""
        return self.metrics.prices

    @prices.setter
    def prices(self, prices: Optional[pd.DataFrame] = None) -> None:
        if prices is None:
            return
        self.metrics.prices = prices
        self.assets = prices.columns

    @property
    def assets(self) -> Optional[pd.Index]:
        """assets"""
        return self.metrics.assets

    @assets.setter
    def assets(self, assets: pd.Index) -> None:
        """assets setter"""
        if self.assets is not None:
            assert self.assets.equals(assets), f"{self.assets} does not equal {assets}"
            return
        self.metrics.assets = assets

    @property
    def num_assets(self) -> int:
        """return number of asset"""
        if self.assets is None:
            return 0
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
            return
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: objectives.expected_return(
                    weights=w, expected_returns=np.array(self.expected_returns)
                )
                - min_return,
            }
        )

    def set_max_return(self, max_return: float = 0.05) -> None:
        """set maximum return constraint"""
        if self.expected_returns is None:
            warnings.warn("unable to set maximum return constraint.")
            warnings.warn("expected returns is null.")
            return
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: max_return
                - objectives.expected_return(
                    weights=w, expected_returns=np.array(self.expected_returns)
                ),
            }
        )

    def set_min_volatility(self, min_volatility: float = 0.05) -> None:
        """set minimum volatility constraint"""
        if self.covariance_matrix is None:
            warnings.warn("unable to set minimum volatility constraint.")
            warnings.warn("covariance matrix is null.")
            return
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: objectives.expected_volatility(
                    weights=w, covariance_matrix=np.array(self.covariance_matrix)
                )
                - min_volatility,
            }
        )

    def set_max_volatility(self, max_volatility: float = 0.05) -> None:
        """set maximum volatility constraint"""
        if self.covariance_matrix is None:
            warnings.warn("unable to set maximum volatility constraint.")
            warnings.warn("covariance matrix is null.")
            return
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: max_volatility
                - objectives.expected_volatility(
                    weights=w, covariance_matrix=np.array(self.covariance_matrix)
                ),
            }
        )

    def set_max_active_weight(self, max_active_weight: float = 0.10) -> None:
        """set maximum active weight against benchmark"""
        if self.weights_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            return
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: max_active_weight
                - np.sum(np.abs(w - np.array(self.weights_bm))),
            }
        )

    def set_max_exante_tracking_error(
        self, max_exante_tracking_error: float = 0.02
    ) -> None:
        """set maximum exante tracking error constraint"""
        if self.weights_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            return
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: max_exante_tracking_error
                - objectives.exante_tracking_error(
                    weights=w,
                    weights_bm=np.array(self.weights_bm),
                    covariance_matrix=np.array(self.covariance_matrix),
                ),
            }
        )

    def set_max_expost_tracking_error(
        self, max_expost_tracking_error: float = 0.02
    ) -> None:
        """set maximum expost tracking error constraint"""
        if self.prices is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            return
        if self.prices_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            return
        itx = self.prices.dropna().index.intersection(self.prices_bm.dropna().index)
        pri_returns_assets = self.prices.loc[itx].pct_change().fillna(0)
        pri_returns_bm = self.prices_bm.loc[itx].pct_change().fillna(0)

        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: max_expost_tracking_error
                - objectives.expost_tracking_error(
                    weights=w,
                    pri_returns_assets=np.array(pri_returns_assets),
                    pri_returns_bm=np.array(pri_returns_bm),
                ),
            }
        )

    def solve(
        self, objective: Callable, extra_constraints: Optional[List[Dict]] = None
    ) -> Optional[pd.Series]:
        """_summary_

        Args:
            objective (Callable): optimization objective.
            extra_constraints (Optional[List[Dict]]): temporary constraints.
                Defaults to None.

        Returns:
            Optional[pd.Series]: optimized weights
        """
        constraints = self.constraints.copy()
        if extra_constraints:
            constraints.extend(extra_constraints)
        problem = minimize(
            fun=objective,
            method="SLSQP",
            constraints=constraints,
            x0=np.ones(shape=self.num_assets) / self.num_assets,
        )

        if problem.success:
            data = problem.x + 1e-16
            return pd.Series(data=data, index=self.assets, name="weights").round(6)
        return None

    def maximized_return(self) -> Optional[pd.Series]:
        """calculate maximum return weights"""
        if self.expected_returns is None:
            warnings.warn("expected_returns must not be none.")
            return
        return self.solve(
            objective=partial(
                objectives.expected_return,
                expected_returns=np.array(self.expected_returns) * -1,
            )
        )

    def minimized_volatility(self) -> Optional[pd.Series]:
        """calculate minimum volatility weights"""
        if self.covariance_matrix is None:
            warnings.warn("covariance_matrix must not be none.")
            return
        return self.solve(
            objective=partial(
                objectives.expected_volatility,
                covariance_matrix=np.array(self.covariance_matrix),
            )
        )

    def maximized_sharpe_ratio(self) -> Optional[pd.Series]:
        """calculate maximum sharpe ratio weights"""
        if self.expected_returns is None or self.covariance_matrix is None:
            warnings.warn("expected_returns and covariance_matrix must not be none.")
            return
        return self.solve(
            objective=partial(
                objectives.expected_sharpe,
                expected_returns=np.array(self.expected_returns * -1),
                covariance_matrix=np.array(self.covariance_matrix),
                risk_free=self.risk_free,
            )
        )

    def hierarchical_equal_risk_contribution(
        self, linkage_method: str = "single"
    ) -> Optional[pd.Series]:
        """calculate herc weights"""
        if self.num_assets <= 2: return self.risk_parity()
        if self.correlation_matrix is None:
            if self.covariance_matrix is not None:
                self.correlation_matrix = cov_to_corr(self.covariance_matrix)
            elif self.prices is not None:
                self.correlation_matrix = estimators.to_correlation_matrix(self.prices)
            else:
                raise ValueError("correlation matrix is none and uncomputable.")
        dist = np.sqrt((1 - self.correlation_matrix).round(5) / 2)
        clusters = linkage(squareform(dist), method=linkage_method)
        sorted_tree = list(to_tree(clusters, rd=False).pre_order())
        cluster_sets = recursive_bisection(sorted_tree)
        if not isinstance(cluster_sets, List):
            cluster_sets = [cluster_sets]
        weights = self.solve(
            objective=lambda w: objectives.l2_norm(
                np.array(
                    [
                        np.sum(
                            objectives.risk_contributions(
                                weights=w,
                                covariance_matrix=np.array(self.covariance_matrix),
                                sub_covariance_matrix_idx=left_idx,
                            )
                        )
                        - np.sum(
                            objectives.risk_contributions(
                                weights=w,
                                covariance_matrix=np.array(self.covariance_matrix),
                                sub_covariance_matrix_idx=right_idx,
                            )
                        )
                        for left_idx, right_idx in cluster_sets
                    ]
                )
            )
        )

        return weights

    def hierarchical_risk_parity(
        self, linkage_method: str = "single"
    ) -> Optional[pd.Series]:
        """calculate herc weights"""
        if self.num_assets <= 2: return self.risk_parity()
        if self.correlation_matrix is None:
            if self.covariance_matrix is not None:
                self.correlation_matrix = cov_to_corr(self.covariance_matrix)
            elif self.prices is not None:
                self.correlation_matrix = estimators.to_correlation_matrix(self.prices)
            else:
                raise ValueError("correlation matrix is none and uncomputable.")

        dist = np.sqrt((1 - self.correlation_matrix).round(5) / 2)
        clusters = linkage(squareform(dist), method=linkage_method)
        sorted_tree = list(to_tree(clusters, rd=False).pre_order())
        cluster_sets = recursive_bisection(sorted_tree)
        if not isinstance(cluster_sets, List):
            cluster_sets = [cluster_sets]
        return self.solve(
            objective=lambda w: objectives.l2_norm(
                np.array(
                    [
                        objectives.expected_volatility(
                            weights=w,
                            covariance_matrix=np.array(self.covariance_matrix),
                            sub_covariance_matrix_idx=left_idx,
                        )
                        - objectives.expected_volatility(
                            weights=w,
                            covariance_matrix=np.array(self.covariance_matrix),
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
            budgets = np.ones(self.num_assets) / self.num_assets
        weights = self.solve(
            objective=lambda w: objectives.l1_norm(
                np.subtract(
                    objectives.risk_contributions(
                        weights=w, covariance_matrix=np.array(self.covariance_matrix)
                    ),
                    np.multiply(
                        budgets,
                        objectives.expected_volatility(
                            weights=w,
                            covariance_matrix=np.array(self.covariance_matrix),
                        ),
                    ),
                )
            )
        )
        return weights

    def inverse_variance(self) -> Optional[pd.Series]:
        """_summary_

        Returns:
            Optional[pd.Series]: _description_
        """
        inv_var_weights = 1 / np.diag(np.array(self.covariance_matrix))
        inv_var_weights /= inv_var_weights.sum()
        return self.solve(
            objective=lambda w: objectives.l1_norm(np.subtract(w, inv_var_weights))
        )

    def uniform_allocation(self) -> Optional[pd.Series]:
        """_summary_

        Returns:
            Optional[pd.Series]: _description_
        """
        target_allocations = np.ones(shape=self.num_assets) / self.num_assets
        return self.solve(
            objective=lambda w: objectives.l1_norm(np.subtract(w, target_allocations))
        )
