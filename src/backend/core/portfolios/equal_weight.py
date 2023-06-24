"""ROBERT"""

from .base import BaseOptimizer
from . import objectives
import numpy as np


class EqualWeight(BaseOptimizer):
    def solve(self) -> pd.Series:
        target_allocations = np.ones(shape=self.num_assets) / self.num_assets
        return self.__solve__(
            objective=lambda w: objectives.l2_norm(np.subtract(w, target_allocations))
        )
