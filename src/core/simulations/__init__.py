from typing import Optional
import numpy as np
import pandas as pd
from .. import metrics


class GeometricBrownianMotion:
    def __init__(
        self,
        prices: pd.DataFrame,
        weights: Optional[pd.Series] = None,
        num_iters: int = 10_000,
        num_years: int = 10,
    ) -> None:
        self.prices = prices
        self.assets = self.prices.columns
        self.num_assets = len(self.assets)
        self.log_returns = metrics.to_log_return(prices=prices.resample("M").last())
        self.weights = weights
        self.num_years = num_years
        self.num_iters = num_iters

    def generate_path(self) -> None:
        mean = self.log_returns.mean()
        std = self.log_returns.std()
        cov = self.log_returns.cov()
        corr_cov = np.linalg.cholesky(cov)
        num_total_iters = 12 * self.num_years * self.num_iters
        z = np.random.normal(
            0, 1, size=(self.num_assets, num_total_iters)
        )

        drift = np.full(())