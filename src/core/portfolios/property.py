"""ROBERT"""
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class BaseProperty:
    @property
    def expected_returns(self) -> Optional[pd.Series]:
        """Get expected returns."""
        try:
            return self._expected_returns
        except AttributeError:
            return None

    @expected_returns.setter
    def expected_returns(self, expected_returns: Optional[pd.Series]) -> None:
        """Set expected returns."""
        if expected_returns is None:
            return
        self._expected_returns = expected_returns
        self.assets = expected_returns.index

    @property
    def covariance_matrix(self) -> Optional[pd.DataFrame]:
        """Get covariance matrix."""
        try:
            return self._covariance_matrix
        except AttributeError:
            return None

    @covariance_matrix.setter
    def covariance_matrix(self, covariance_matrix: Optional[pd.DataFrame]) -> None:
        """Set covariance matrix."""
        if covariance_matrix is not None:
            self._covariance_matrix = covariance_matrix
            self.assets = covariance_matrix.index
            self.assets = covariance_matrix.columns

    @property
    def correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get correlation matrix."""
        try:
            return self._correlation_matrix
        except AttributeError:
            return None

    @correlation_matrix.setter
    def correlation_matrix(self, correlation_matrix: Optional[pd.DataFrame]) -> None:
        """Set correlation matrix."""
        if correlation_matrix is not None:
            self._correlation_matrix = correlation_matrix
            self.assets = correlation_matrix.index
            self.assets = correlation_matrix.columns

    @property
    def prices(self) -> Optional[pd.DataFrame]:
        """Get asset prices."""
        try:
            return self._prices
        except AttributeError:
            return None

    @prices.setter
    def prices(self, prices: Optional[pd.DataFrame]) -> None:
        """Set asset prices."""
        if prices is None:
            return
        self._prices = prices

    @property
    def num_assets(self) -> int:
        """return number of asset"""
        if self.assets is None:
            return 0
        return len(self.assets)

    @property
    def risk_free(self) -> Optional[float]:
        try:
            return self._risk_free
        except AttributeError:
            return None

    @risk_free.setter
    def risk_free(self, risk_free: float) -> None:
        self._risk_free = risk_free

    @property
    def factors(self) -> Optional[pd.Series]:
        """Get asset prices."""
        try:
            return self._factors
        except AttributeError:
            return None

    @factors.setter
    def factors(self, factors: Optional[pd.Series]) -> None:
        if factors is None or not isinstance(factors, pd.Series):
            return
        self._factors = factors.reindex(index=self.assets, fill_value=0).fillna(0)

    @property
    def weights_bm(self) -> Optional[pd.Series]:
        try:
            return self._weights_bm
        except AttributeError:
            return None

    @weights_bm.setter
    def weights_bm(self, weights_bm: Optional[pd.Series]) -> None:
        if weights_bm is not None:
            weights_bm = weights_bm.reindex(self.assets, fill_value=0.0).fillna(0)
            self._weights_bm = weights_bm

    @property
    def prices_bm(self) -> Optional[pd.Series]:
        try:
            return self._prices_bm
        except AttributeError:
            return None

    @prices_bm.setter
    def prices_bm(self, prices_bm: Optional[pd.Series]) -> None:
        if prices_bm is not None and self.prices is not None:
            prices_bm = prices_bm.reindex(self.prices.index).ffill()
            self._prices_bm = prices_bm
