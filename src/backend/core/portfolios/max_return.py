"""ROBERT"""
import warnings
from functools import partial
import pandas as pd
from .base import BaseOptimizer
from . import objectives


class MaxReturn(BaseOptimizer):
    def solve(self) -> pd.Series:
        """calculate maximum return weights"""
        if self.expected_returns is None:
            warnings.warn("expected_returns must not be none.")
            raise ValueError("expected_returns must not be none.")
        return self.__solve__(
            objective=partial(
                objectives.expected_return,
                expected_returns=self.expected_returns.values * -1,
            )
        )
