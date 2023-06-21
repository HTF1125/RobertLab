"""ROBERT"""
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class OptimizerMetrics:
    def __init__(self) -> None:
        self.prices: Optional[pd.DataFrame] = None
        self.expected_returns: Optional[pd.Series] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.assets: Optional[pd.Index] = None
        self.risk_free: float = 0.0


class BaseProperty:
    def __init__(self) -> None:
        self.data = OptimizerMetrics()

    @property
    def expected_returns(self) -> Optional[pd.Series]:
        """Get expected returns."""
        return self.data.expected_returns

    @expected_returns.setter
    def expected_returns(self, expected_returns: Optional[pd.Series] = None) -> None:
        """Set expected returns."""
        if expected_returns is not None:
            self.data.expected_returns = expected_returns
            self.assets = expected_returns.index

    @property
    def covariance_matrix(self) -> Optional[pd.DataFrame]:
        """Get covariance matrix."""
        return self.data.covariance_matrix

    @covariance_matrix.setter
    def covariance_matrix(
        self, covariance_matrix: Optional[pd.DataFrame] = None
    ) -> None:
        """Set covariance matrix."""
        if covariance_matrix is not None:
            self.data.covariance_matrix = covariance_matrix
            self.assets = covariance_matrix.index
            self.assets = covariance_matrix.columns

    @property
    def correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get correlation matrix."""
        return self.data.correlation_matrix

    @correlation_matrix.setter
    def correlation_matrix(
        self, correlation_matrix: Optional[pd.DataFrame] = None
    ) -> None:
        """Set correlation matrix."""
        if correlation_matrix is not None:
            self.data.correlation_matrix = correlation_matrix
            self.assets = correlation_matrix.index
            self.assets = correlation_matrix.columns

    @property
    def prices(self) -> Optional[pd.DataFrame]:
        """Get asset prices."""
        return self.data.prices

    @prices.setter
    def prices(self, prices: Optional[pd.DataFrame] = None) -> None:
        """Set asset prices."""
        if prices is None:
            return
        self.data.prices = prices

    @property
    def num_assets(self) -> int:
        """return number of asset"""
        if self.assets is None:
            return 0
        return len(self.assets)

    @property
    def risk_free(self) -> float:
        return self.data.risk_free

    @risk_free.setter
    def risk_free(self, risk_free: float) -> None:
        self.data.risk_free = risk_free