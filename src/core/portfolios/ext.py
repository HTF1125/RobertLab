"""ROBERT"""

from typing import Optional, List
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from src.core import metrics
from .base import Optimizer



class MinVolatility(Optimizer):
    def solve(self) -> pd.Series:
        """calculate minimum volatility weights"""
        return self.__solve__(objective=self.expected_volatility)


class MinCorrelation(Optimizer):
    def solve(self) -> pd.Series:
        """calculate minimum correlation weights"""
        return self.__solve__(objective=self.expected_correlation)


class RiskParity(Optimizer):
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


class Hierarchical(Optimizer):
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
                self.correlation_matrix = metrics.cov_to_corr(self.covariance_matrix)
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


class InverseVariance(Optimizer):
    def solve(self) -> pd.Series:
        inv_var_weights = 1 / np.diag(np.array(self.covariance_matrix))
        inv_var_weights /= inv_var_weights.sum()
        return self.__solve__(
            objective=lambda w: self.l1_norm(np.subtract(w, inv_var_weights))
        )


class MaxReturn(Optimizer):
    def solve(self) -> pd.Series:
        """calculate maximum return weights"""
        return self.__solve__(objective=lambda w: -self.expected_return(w))


class MaxSharpe(Optimizer):
    def solve(self) -> pd.Series:
        """calculate maximum sharpe ratio weights"""
        return self.__solve__(objective=lambda w: -self.expected_sharpe(w))


class EqualWeight(Optimizer):
    def solve(self) -> pd.Series:
        target_allocations = np.ones(shape=self.num_assets) / self.num_assets
        return self.__solve__(
            objective=lambda w: self.l2_norm(np.subtract(w, target_allocations))
        )
