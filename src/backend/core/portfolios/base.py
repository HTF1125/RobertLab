"""ROBERT"""
import warnings
from abc import abstractmethod
from typing import Optional, Callable, Dict, List, Tuple, Any
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
from .. import metrics
from ..metrics import cov_to_corr
from .objectives import Objectives


class PortfolioOptimizer(Objectives):
    @classmethod
    def from_prices(
        cls, prices: pd.DataFrame, span: Optional[int] = None, **kwargs
    ) -> "PortfolioOptimizer":
        return cls(
            expected_returns=metrics.to_expected_returns(prices=prices),
            covariance_matrix=metrics.to_covariance_matrix(prices=prices, span=span),
            correlation_matrix=metrics.to_correlation_matrix(prices=prices, span=span),
            **kwargs,
        )

    def __init__(
        self,
        expected_returns: Optional[pd.Series] = None,
        covariance_matrix: Optional[pd.DataFrame] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        prices: Optional[pd.DataFrame] = None,
        factors: Optional[pd.Series] = None,
        risk_free: float = 0.0,
        prices_bm: Optional[pd.Series] = None,
        weights_bm: Optional[pd.Series] = None,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        sum_weight: float = 1.0,
        min_active_weight: Optional[float] = None,
        max_active_weight: Optional[float] = None,
        min_return: Optional[float] = None,
        max_return: Optional[float] = None,
        min_volatility: Optional[float] = None,
        max_volatility: Optional[float] = None,
        min_exante_tracking_error: Optional[float] = None,
        max_exante_tracking_error: Optional[float] = None,
        min_expost_tracking_error: Optional[float] = None,
        max_expost_tracking_error: Optional[float] = None,
    ) -> None:
        """init"""
        super().__init__()
        self.constraints = {}
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.correlation_matrix = correlation_matrix
        self.prices = prices
        self.risk_free = risk_free
        self.prices_bm = prices_bm
        self.weights_bm = weights_bm
        self.factors = factors
        self.exp = {}
        self.set_sum_weight(sum_weight)
        self.set_min_weight(min_weight)
        self.set_max_weight(max_weight)
        self.set_min_factor_percentile()
        if min_active_weight is not None:
            self.set_min_active_weight(min_active_weight)
        if max_active_weight is not None:
            self.set_max_active_weight(max_active_weight)
        if min_return is not None:
            self.set_min_port_return(min_return)
        if max_return is not None:
            self.set_max_port_return(max_return)
        if min_volatility is not None:
            self.set_min_port_risk(min_volatility)
        if max_volatility is not None:
            self.set_max_port_risk(max_volatility)
        if min_exante_tracking_error is not None:
            self.set_min_exante_tracking_error(min_exante_tracking_error)
        if max_exante_tracking_error is not None:
            self.set_max_exante_tracking_error(max_exante_tracking_error)
        if min_expost_tracking_error is not None:
            self.set_min_expost_tracking_error(min_expost_tracking_error)
        if max_expost_tracking_error is not None:
            self.set_max_expost_tracking_error(max_expost_tracking_error)

    def set_sum_weight(self, sum_weight: float) -> None:
        """set summation weights constriant"""
        self.constraints["sum_weight"] = {
            "type": "eq",
            "fun": lambda w: np.sum(w) - sum_weight,
        }

    def set_min_weight(self, min_weight: float) -> None:
        """set minimum weight constraint"""
        self.constraints["min_weight"] = {
            "type": "ineq",
            "fun": lambda w: w - min_weight,
        }

    def set_max_weight(self, max_weight: float) -> None:
        """set minimum weight constraint"""
        self.constraints["max_weight"] = {
            "type": "ineq",
            "fun": lambda w: max_weight - w,
        }

    def set_min_port_return(self, min_port_return: float) -> None:
        """set minimum return constraint"""
        if self.expected_returns is None:
            warnings.warn("unable to set minimum return constraint.")
            warnings.warn("expected returns is null.")
            return
        self.constraints["min_port_return"] = {
            "type": "ineq",
            "fun": lambda w: self.expected_return(weights=w) - min_port_return,
        }

    def set_max_port_return(self, max_port_return: float) -> None:
        """set maximum return constraint"""
        if self.expected_returns is None:
            warnings.warn("unable to set maximum return constraint.")
            warnings.warn("expected returns is null.")
            return
        self.constraints["max_port_return"] = {
            "type": "ineq",
            "fun": lambda w: max_port_return - self.expected_return(weights=w),
        }

    def set_min_port_risk(self, min_port_risk: float) -> None:
        """set minimum volatility constraint"""
        if self.covariance_matrix is None:
            warnings.warn("unable to set minimum volatility constraint.")
            warnings.warn("covariance matrix is null.")
            return
        self.constraints["min_port_risk"] = {
            "type": "ineq",
            "fun": lambda w: self.expected_volatility(weights=w) - min_port_risk,
        }

    def set_max_port_risk(self, max_port_risk: float) -> None:
        """set maximum volatility constraint"""
        if self.covariance_matrix is None:
            warnings.warn("unable to set maximum volatility constraint.")
            warnings.warn("covariance matrix is null.")
            return
        self.constraints["max_port_risk"] = {
            "type": "ineq",
            "fun": lambda w: max_port_risk - self.expected_volatility(weights=w),
        }

    def set_min_active_weight(self, min_active_weight: float) -> None:
        """set maximum active weight against benchmark"""
        if self.weights_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            return
        self.constraints["min_active_weight"] = {
            "type": "ineq",
            "fun": lambda w: np.sum(np.abs(w - np.array(self.weights_bm)))
            - min_active_weight,
        }

    def set_max_active_weight(self, max_active_weight: float = 0.10) -> None:
        """set maximum active weight against benchmark"""
        if self.weights_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            return
        self.constraints["max_active_weight"] = {
            "type": "ineq",
            "fun": lambda w: max_active_weight
            - np.sum(np.abs(w - np.array(self.weights_bm))),
        }

    def set_min_exante_tracking_error(
        self, min_exante_tracking_error: float = 0.02
    ) -> None:
        """set maximum exante tracking error constraint"""
        if self.weights_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            return
        self.constraints["min_exante_tracking_error"] = {
            "type": "ineq",
            "fun": lambda w: self.exante_tracking_error(
                weights=w, weights_bm=np.array(self.weights_bm)
            )
            - min_exante_tracking_error,
        }

    def set_max_exante_tracking_error(
        self, max_exante_tracking_error: float = 0.02
    ) -> None:
        """set maximum exante tracking error constraint"""
        if self.weights_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            return
        self.constraints["max_exante_tracking_error"] = {
            "type": "ineq",
            "fun": lambda w: max_exante_tracking_error
            - self.exante_tracking_error(
                weights=w, weights_bm=np.array(self.weights_bm)
            ),
        }

    def set_min_expost_tracking_error(
        self, min_expost_tracking_error: float = 0.02
    ) -> None:
        """set maximum expost tracking error constraint"""
        if self.prices is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            raise ValueError("prices must not be none.")
        if self.prices_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            raise ValueError("prices_bm must not be none.")
        itx = self.prices.dropna().index.intersection(self.prices_bm.dropna().index)
        pri_returns_assets = self.prices.loc[itx].pct_change().fillna(0)
        pri_returns_bm = self.prices_bm.loc[itx].pct_change().fillna(0)

        self.constraints["min_expost_tracking_error"] = {
            "type": "ineq",
            "fun": lambda w: self.expost_tracking_error(
                weights=w,
                pri_returns_assets=np.array(pri_returns_assets),
                pri_returns_bm=np.array(pri_returns_bm),
            )
            - min_expost_tracking_error,
        }

    def set_max_expost_tracking_error(
        self, max_expost_tracking_error: float = 0.02
    ) -> None:
        """set maximum expost tracking error constraint"""
        if self.prices is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            raise ValueError("prices must not be none.")
        if self.prices_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            raise ValueError("prices_bm must not be none.")
        itx = self.prices.dropna().index.intersection(self.prices_bm.dropna().index)
        pri_returns_assets = self.prices.loc[itx].pct_change().fillna(0)
        pri_returns_bm = self.prices_bm.loc[itx].pct_change().fillna(0)

        self.constraints["max_expost_tracking_error"] = {
            "type": "ineq",
            "fun": lambda w: max_expost_tracking_error
            - self.expost_tracking_error(
                weights=w,
                pri_returns_assets=np.array(pri_returns_assets),
                pri_returns_bm=np.array(pri_returns_bm),
            ),
        }

    def set_min_factor_percentile(self) -> None:
        if self.factors is not None:
            lbound = self.factors.quantile(0.7)
            self.constraints["min_factor"] = {
                "type": "ineq",
                "fun": lambda w: np.dot(w, np.array(self.factors)) - lbound,
            }

    def set_specific_constraints(
        self, specific_constraints: List[Dict[str, Any]]
    ) -> "PortfolioOptimizer":
        for specific_constraint in specific_constraints:
            self.set_specific_constraint(**specific_constraint)
        return self

    def set_specific_constraint(self, assets: List, bounds: Tuple) -> "PortfolioOptimizer":
        assert self.assets is not None
        specific_assets = np.in1d(self.assets.values, assets)
        l_bound, u_bound = bounds
        if l_bound is not None:
            self.constraints["min_" + str(assets)] = {
                "type": "ineq",
                "fun": lambda w: np.dot(w, specific_assets) - l_bound,
            }
        if u_bound is not None:
            self.constraints["max_" + str(assets)] = {
                "type": "ineq",
                "fun": lambda w: u_bound - np.dot(w, specific_assets),
            }
        return self

    def __solve__(
        self, objective: Callable, extra_constraints: Optional[List[Dict]] = None
    ) -> pd.Series:
        """_summary_

        Args:
            objective (Callable): optimization objective.
            extra_constraints (Optional[List[Dict]]): temporary constraints.
                Defaults to None.

        Returns:
            pd.Series: optimized weights
        """
        constraints = list(self.constraints.values())
        if extra_constraints:
            constraints.extend(extra_constraints)
        problem = minimize(
            fun=objective,
            method="SLSQP",
            constraints=constraints,
            x0=np.ones(shape=self.num_assets) / self.num_assets,
        )
        if problem.success:
            weights = problem.x + 1e-16
            if self.expected_returns is not None:
                self.exp["expected_return"] = self.expected_return(weights)
            if self.covariance_matrix is not None:
                self.exp["expected_volatility"] = self.expected_volatility(weights)
            weights = pd.Series(data=weights, index=self.assets, name="weights").round(
                6
            )
            weights = weights[weights != 0.0]
            return weights
        raise ValueError("Portoflio Optimization Failed.")

    @abstractmethod
    def solve(self):
        raise ValueError(
            "Must implement `solve` method for subclasses of BaseOptimizer."
        )


class MinVolatility(PortfolioOptimizer):
    def solve(self) -> pd.Series:
        """calculate minimum volatility weights"""
        return self.__solve__(objective=self.expected_volatility)


class MinCorrelation(PortfolioOptimizer):
    def solve(self) -> pd.Series:
        """calculate minimum correlation weights"""
        return self.__solve__(objective=self.expected_correlation)


class RiskParity(PortfolioOptimizer):
    def solve(self, budgets: Optional[np.ndarray] = None) -> pd.Series:
        if budgets is None:
            budgets = np.ones(self.num_assets) / self.num_assets
        weights = self.__solve__(
            objective=lambda w: self.l2_norm(
                np.subtract(
                    self.risk_contributions(weights=w),
                    np.multiply(
                        budgets,
                        self.expected_volatility(weights=w),
                    ),
                )
            )
        )
        return weights


class Hierarchical(PortfolioOptimizer):
    def recursive_bisection(self, sorted_tree):
        """_summary_

        Args:
            sorted_tree (_type_): _description_

        Returns:
            List[Tuple[List[int], List[int]]]: _description_
        """

        left = sorted_tree[0 : int(len(sorted_tree) / 2)]
        right = sorted_tree[int(len(sorted_tree) / 2) :]

        # if len(sorted_tree) <= 3:
        #     return [(left, right)]

        cache = [(left, right)]
        if len(left) > 2:
            cache.extend(self.recursive_bisection(left))
        if len(right) > 2:
            cache.extend(self.recursive_bisection(right))
        return cache


class HRP(Hierarchical):
    def solve(self, linkage_method: str = "single") -> pd.Series:
        if self.correlation_matrix is None:
            if self.prices is not None:
                self.correlation_matrix = metrics.to_correlation_matrix(self.prices)
            elif self.covariance_matrix is not None:
                self.correlation_matrix = cov_to_corr(self.covariance_matrix)
            else:
                raise ValueError("correlation matrix is none and uncomputable.")

        dist = np.sqrt((1 - self.correlation_matrix).round(5) / 2)
        clusters = linkage(squareform(dist), method=linkage_method)
        sorted_tree = list(to_tree(clusters, rd=False).pre_order())
        cluster_sets = self.recursive_bisection(sorted_tree)
        if not isinstance(cluster_sets, List):
            cluster_sets = [cluster_sets]
        return self.__solve__(
            objective=lambda w: self.l2_norm(
                np.array(
                    [
                        self.expected_volatility(
                            weights=w,
                            sub_covariance_matrix_idx=left_idx,
                        )
                        - self.expected_volatility(
                            weights=w,
                            sub_covariance_matrix_idx=right_idx,
                        )
                        for left_idx, right_idx in cluster_sets
                    ]
                )
            )
        )


class HERC(Hierarchical):
    def solve(self, linkage_method: str = "single") -> pd.Series:
        """calculate herc weights"""

        if self.correlation_matrix is None:
            if self.covariance_matrix is not None:
                self.correlation_matrix = metrics.cov_to_corr(self.covariance_matrix)
            elif self.prices is not None:
                self.correlation_matrix = metrics.to_correlation_matrix(self.prices)
            else:
                raise ValueError("correlation matrix is none and uncomputable.")
        dist = np.sqrt((1 - self.correlation_matrix).round(5) / 2)
        clusters = linkage(squareform(dist), method=linkage_method)
        sorted_tree = list(to_tree(clusters, rd=False).pre_order())
        cluster_sets = self.recursive_bisection(sorted_tree)
        weights = self.__solve__(
            objective=lambda w: self.l2_norm(
                np.array(
                    [
                        np.sum(
                            self.risk_contributions(
                                weights=w,
                                sub_covariance_matrix_idx=left_idx,
                            )
                        )
                        - np.sum(
                            self.risk_contributions(
                                weights=w,
                                sub_covariance_matrix_idx=right_idx,
                            )
                        )
                        for left_idx, right_idx in cluster_sets
                    ]
                )
            )
        )
        return weights


class InverseVariance(PortfolioOptimizer):
    def solve(self) -> pd.Series:
        inv_var_weights = 1 / np.diag(np.array(self.covariance_matrix))
        inv_var_weights /= inv_var_weights.sum()
        return self.__solve__(
            objective=lambda w: self.l1_norm(np.subtract(w, inv_var_weights))
        )


class MaxReturn(PortfolioOptimizer):
    def solve(self) -> pd.Series:
        """calculate maximum return weights"""
        return self.__solve__(objective=lambda w: -self.expected_return(w))


class MaxSharpe(PortfolioOptimizer):
    def solve(self) -> pd.Series:
        """calculate maximum sharpe ratio weights"""
        return self.__solve__(objective=lambda w: -self.expected_sharpe(w))


class EqualWeight(PortfolioOptimizer):
    def solve(self) -> pd.Series:
        target_allocations = np.ones(shape=self.num_assets) / self.num_assets
        return self.__solve__(
            objective=lambda w: self.l2_norm(np.subtract(w, target_allocations))
        )
