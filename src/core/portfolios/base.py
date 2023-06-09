"""ROBERT"""
import sys
from abc import abstractmethod
from typing import Optional, Callable, Dict, List, Tuple, Any, Type, Union
from scipy.optimize import minimize
from scipy.stats import percentileofscore
import numpy as np
import pandas as pd
from src.core import metrics
from src.core.portfolios.constraints import Constraints


class Portfolio(Constraints):
    optimizer_metrics = {}

    def __init__(self):
        super().__init__()
        self.min_weight = 0.0
        self.max_weight = 1.0
        self.min_leverage = 0.0
        self.max_leverage = 1.0
        self.risk_free = 0.0

    @classmethod
    def from_prices(
        cls, prices: pd.DataFrame, span: Optional[int] = None, **kwargs
    ) -> "Portfolio":
        expected_returns = metrics.to_expected_returns(prices)
        covariance_matrix = metrics.to_covariance_matrix(prices, span=span)
        correlation_matrix = metrics.to_correlation_matrix(prices, span=span)
        return cls().new(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
            **kwargs,
        )

    def new(
        self,
        expected_returns: Optional[pd.Series] = None,
        covariance_matrix: Optional[pd.DataFrame] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        min_leverage: float = 0.0,
        max_leverage: float = 0.0,
        prices: Optional[pd.DataFrame] = None,
        factor: Optional[pd.Series] = None,
        risk_free: float = 0.0,
        prices_bm: Optional[pd.Series] = None,
        weights_bm: Optional[pd.Series] = None,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
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
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
        min_factor_percentile: float = 0.2,
    ) -> "Portfolio":
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.correlation_matrix = correlation_matrix
        self.prices = prices
        self.risk_free = risk_free
        self.prices_bm = prices_bm
        self.weights_bm = weights_bm
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_active_weight = min_active_weight
        self.max_active_weight = max_active_weight
        self.min_return = min_return
        self.max_return = max_return
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        self.min_exante_tracking_error = min_exante_tracking_error
        self.max_exante_tracking_error = max_exante_tracking_error
        self.min_expost_tracking_error = min_expost_tracking_error
        self.max_expost_tracking_error = max_expost_tracking_error
        self.min_factor_percentile = min_factor_percentile
        self.factor = factor
        if specific_constraints is not None:
            self.set_specific_constraints(specific_constraints)
        self.min_factor_percentile = 0.2
        return self

    def set_specific_constraints(
        self, specific_constraints: List[Dict[str, Any]]
    ) -> "Portfolio":
        for specific_constraint in specific_constraints:
            self.set_specific_constraint(**specific_constraint)
        return self

    def set_specific_constraint(
        self, assets: List[str], bounds: Tuple[Optional[float], Optional[float]]
    ) -> "Portfolio":
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
                self.optimizer_metrics["expected_return"] = self.expected_return(
                    weights
                )
            if self.covariance_matrix is not None:
                self.optimizer_metrics[
                    "expected_volatility"
                ] = self.expected_volatility(weights)
            weights = pd.Series(data=weights, index=self.assets, name="weights").round(
                6
            )
            return weights

        if self.factor is not None and self.min_factor_percentile is not None:
            self.min_factor_percentile -= 0.05
            if self.min_factor_percentile < 0.0:
                raise ValueError(
                    "Portfolio Optimization Failed. After adjusting factor percentile."
                )
            return self.__solve__(
                objective=objective, extra_constraints=extra_constraints
            )

        print(self.constraints)
        print(dir(self))


        raise ValueError("Portfolio Optimization Failed.")

    @abstractmethod
    def solve(self):
        raise NotImplementedError(
            "Must implement `solve` method for subclasses of Optimizer."
        )

    @property
    def min_factor_percentile(self) -> Optional[float]:
        return self._min_factor_percentile

    @min_factor_percentile.setter
    def min_factor_percentile(self, min_factor_percentile: Optional[float]) -> None:
        if min_factor_percentile is None:
            return
        self._min_factor_percentile = min_factor_percentile

        if self.factor is not None:
            if self.weights_bm is not None:
                w = self.weights_bm.copy()
            else:
                w = self.solve()

            factor = np.dot(w, np.array(self.factor))
            self.constraints["min_factor_percentile"] = {
                "type": "ineq",
                "fun": lambda w: np.dot(w, np.array(self.factor))
                - (factor + self.min_factor_percentile),
            }
