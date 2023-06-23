"""ROBERT"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseOptimizer
from . import objectives


class RiskParity(BaseOptimizer):
    def solve(self, budgets: Optional[np.ndarray] = None) -> pd.Series:
        if budgets is None:
            budgets = np.ones(self.num_assets) / self.num_assets
        weights = self.__solve__(
            objective=lambda w: objectives.l2_norm(
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
