import logging
import warnings
from typing import Optional, Callable, Dict, List, Tuple, Union, Any
from functools import partial
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
from . import objectives
from .. import metrics
from ..metrics import cov_to_corr


logger = logging.getLogger(__name__)


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
    def from_prices(
        cls,
        prices: pd.DataFrame,
        span: Optional[int] = None,
        risk_free: float = 0.0,
        prices_bm: Optional[pd.Series] = None,
        weights_bm: Optional[pd.Series] = None,
        weight_bounds: Optional[Tuple[Optional[float], Optional[float]]] = (0.0, 1.0),
        return_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
        risk_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
        active_weight_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
        exante_tracking_error_bounds: Optional[
            Tuple[Optional[float], Optional[float]]
        ] = None,
        expost_tracking_error_bounds: Optional[
            Tuple[Optional[float], Optional[float]]
        ] = None,
    ) -> "Optimizer":
        """_summary_

        Args:
            prices (pd.DataFrame): price of assets.

        Returns:
            Optimizer: initialized optimizer class.
        """
        return cls(
            expected_returns=metrics.to_expected_returns(prices=prices),
            covariance_matrix=metrics.to_covariance_matrix(prices=prices, span=span),
            correlation_matrix=metrics.to_correlation_matrix(prices=prices, span=span),
            risk_free=risk_free,
            prices_bm=prices_bm,
            weights_bm=weights_bm,
            weight_bounds=weight_bounds,
            return_bounds=return_bounds,
            risk_bounds=risk_bounds,
            active_weight_bounds=active_weight_bounds,
            exante_tracking_error_bounds=exante_tracking_error_bounds,
            expost_tracking_error_bounds=expost_tracking_error_bounds,
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
        weight_bounds: Optional[Tuple[Optional[float], Optional[float]]] = (0.0, 1.0),
        return_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
        risk_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
        active_weight_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
        exante_tracking_error_bounds: Optional[
            Tuple[Optional[float], Optional[float]]
        ] = None,
        expost_tracking_error_bounds: Optional[
            Tuple[Optional[float], Optional[float]]
        ] = None,
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
        self.set_sum_weight(sum_weight=1)

        if weight_bounds is not None:
            min_weight, max_weight = weight_bounds
            if min_weight is not None:
                self.set_min_weight(min_weight=min_weight)
            if max_weight is not None:
                self.set_max_weight(max_weight=max_weight)

        if return_bounds is not None:
            min_return, max_return = return_bounds
            if min_return is not None:
                self.set_min_return(min_return)
            if max_return is not None:
                self.set_max_return(max_return)

        if risk_bounds is not None:
            min_risk, max_risk = risk_bounds
            if min_risk is not None:
                self.set_min_risk(min_risk)
            if max_risk is not None:
                self.set_max_risk(max_risk)
        if active_weight_bounds:
            min_active_weight, max_active_weight = active_weight_bounds
            if min_active_weight is not None:
                self.set_min_active_weight(min_active_weight)
            if max_active_weight is not None:
                self.set_max_active_weight(max_active_weight)

        if exante_tracking_error_bounds:
            (
                min_exante_tracking_error,
                max_exante_tracking_error,
            ) = exante_tracking_error_bounds
            if min_exante_tracking_error is not None:
                self.set_min_exante_tracking_error(min_exante_tracking_error)
            if max_exante_tracking_error is not None:
                self.set_max_exante_tracking_error(max_exante_tracking_error)

        if expost_tracking_error_bounds:
            (
                min_expost_tracking_error,
                max_expost_tracking_error,
            ) = expost_tracking_error_bounds
            if min_expost_tracking_error is not None:
                self.set_min_expost_tracking_error(min_expost_tracking_error)
            if max_expost_tracking_error is not None:
                self.set_max_expost_tracking_error(max_expost_tracking_error)

    def set_risk_bounds(
        self, bounds: Union[float, Tuple[Optional[float], Optional[float]]]
    ) -> "Optimizer":
        if isinstance(bounds, float):
            self.constraints.append(
                {
                    "type": "eq",
                    "fun": lambda w: objectives.expected_volatility(
                        weights=w, covariance_matrix=np.array(self.covariance_matrix)
                    )
                    - bounds,
                }
            )
            return self
        _min, _max = bounds

        if _min:
            self.constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: objectives.expected_volatility(
                        weights=w, covariance_matrix=np.array(self.covariance_matrix)
                    )
                    - _min,
                }
            )
        if _max:
            self.constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: _max
                    - objectives.expected_volatility(
                        weights=w, covariance_matrix=np.array(self.covariance_matrix)
                    ),
                }
            )

        return self

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

    def set_min_risk(self, min_risk: float = 0.05) -> None:
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
                - min_risk,
            }
        )

    def set_max_risk(self, max_risk: float = 0.05) -> None:
        """set maximum volatility constraint"""
        if self.covariance_matrix is None:
            warnings.warn("unable to set maximum volatility constraint.")
            warnings.warn("covariance matrix is null.")
            return
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: max_risk
                - objectives.expected_volatility(
                    weights=w, covariance_matrix=np.array(self.covariance_matrix)
                ),
            }
        )

    def set_min_active_weight(self, min_active_weight: float = 0.10) -> None:
        """set maximum active weight against benchmark"""
        if self.weights_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            return
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: np.sum(np.abs(w - np.array(self.weights_bm)))
                - min_active_weight,
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

    def set_min_exante_tracking_error(
        self, min_exante_tracking_error: float = 0.02
    ) -> None:
        """set maximum exante tracking error constraint"""
        if self.weights_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            return
        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: objectives.exante_tracking_error(
                    weights=w,
                    weights_bm=np.array(self.weights_bm),
                    covariance_matrix=np.array(self.covariance_matrix),
                )
                - min_exante_tracking_error,
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

        self.constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: objectives.expost_tracking_error(
                    weights=w,
                    pri_returns_assets=np.array(pri_returns_assets),
                    pri_returns_bm=np.array(pri_returns_bm),
                )
                - min_expost_tracking_error,
            }
        )

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

    def set_factor_constraints(
        self,
        values: pd.Series,
        bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
    ) -> "Optimizer":
        l_bound, u_bound = bounds

        if l_bound is not None:
            self.constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: np.dot(
                        w, values.reindex(self.assets, fill_value=0) - l_bound
                    ),
                }
            )

        if u_bound is not None:
            self.constraints.append(
                {
                    "type": "eq",
                    "fun": lambda w: u_bound
                    - np.dot(w, values.reindex(self.assets, fill_value=0)),
                }
            )

        return self

    def set_specific_constraints(
        self, specific_constraints: List[Dict[str, Any]]
    ) -> "Optimizer":
        for specific_constraint in specific_constraints:
            self.set_specific_constraint(**specific_constraint)
        return self

    def set_specific_constraint(self, assets: List, bounds: Tuple) -> "Optimizer":
        assert self.assets is not None
        specific_assets = np.in1d(self.assets.values, assets)
        l_bound, u_bound = bounds
        if l_bound is not None:
            self.constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: np.dot(w, specific_assets) - l_bound,
                }
            )
        if u_bound is not None:
            self.constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: u_bound - np.dot(w, specific_assets),
                }
            )
        return self

    def solve(
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
        return pd.Series(dtype=float)

    def maximized_return(self) -> pd.Series:
        """calculate maximum return weights"""
        if self.expected_returns is None:
            warnings.warn("expected_returns must not be none.")
            raise ValueError("expected_returns must not be none.")
        return self.solve(
            objective=partial(
                objectives.expected_return,
                expected_returns=np.array(self.expected_returns) * -1,
            )
        )

    def minimized_volatility(self) -> pd.Series:
        """calculate minimum volatility weights"""
        if self.covariance_matrix is None:
            warnings.warn("covariance_matrix must not be none.")
            raise ValueError("covariance_matrix must not be none.")

        return self.solve(
            objective=partial(
                objectives.expected_volatility,
                covariance_matrix=np.array(self.covariance_matrix),
            )
        )

    def minimized_correlation(self) -> pd.Series:
        """calculate minimum correlation weights"""

        return self.solve(
            objective=partial(
                objectives.expected_correlation,
                correlation_matrix=np.array(self.correlation_matrix),
            )
        )

    def maximized_sharpe_ratio(self) -> pd.Series:
        """calculate maximum sharpe ratio weights"""
        if self.expected_returns is None or self.covariance_matrix is None:
            warnings.warn("expected_returns and covariance_matrix must not be none.")
            raise ValueError("expected_returns must not be none.")

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
    ) -> pd.Series:
        """calculate herc weights"""
        if self.num_assets <= 2:
            return self.risk_parity()
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

    def hierarchical_risk_parity(self, linkage_method: str = "single") -> pd.Series:
        """calculate herc weights"""
        if self.num_assets <= 2:
            return self.risk_parity()
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

    def risk_parity(self, budgets: Optional[np.ndarray] = None) -> pd.Series:
        """_summary_

        Returns:
            pd.Series: _description_
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

    def inverse_variance(self) -> pd.Series:
        """_summary_

        Returns:
            pd.Series: _description_
        """
        inv_var_weights = 1 / np.diag(np.array(self.covariance_matrix))
        inv_var_weights /= inv_var_weights.sum()
        return self.solve(
            objective=lambda w: objectives.l1_norm(np.subtract(w, inv_var_weights))
        )

    def uniform_allocation(self) -> pd.Series:
        """_summary_

        Returns:
            pd.Series: _description_
        """
        target_allocations = np.ones(shape=self.num_assets) / self.num_assets
        return self.solve(
            objective=lambda w: objectives.l2_norm(np.subtract(w, target_allocations))
        )
