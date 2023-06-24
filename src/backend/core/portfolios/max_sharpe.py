"""ROBERT"""
import warnings
from functools import partial
import numpy as np
import pandas as pd
from .base import BaseOptimizer
from . import objectives


class MaxSharpe(BaseOptimizer):
    def solve(self) -> pd.Series:
        """calculate maximum sharpe ratio weights"""
        if self.expected_returns is None or self.covariance_matrix is None:
            warnings.warn("expected_returns and covariance_matrix must not be none.")
            raise ValueError("expected_returns must not be none.")

        return self.__solve__(
            objective=partial(
                objectives.expected_sharpe,
                expected_returns=np.array(self.expected_returns * -1),
                covariance_matrix=np.array(self.covariance_matrix),
                risk_free=self.risk_free,
            )
        )
