"""ROBERT"""
import warnings
from typing import Optional
import numpy as np
from .objectives import Objectives


class Constraints(Objectives):
    def __init__(self) -> None:
        self.constraints = {}

    @property
    def min_leverage(self) -> float:
        return self._min_leverage

    @min_leverage.setter
    def min_leverage(self, min_leverage: float) -> None:
        self._min_leverage = min_leverage
        self.constraints["min_leverage"] = {
            "type": "ineq",
            "fun": lambda w: np.sum(w) - (min_leverage + 1.0),
        }

    @property
    def max_leverage(self) -> float:
        return self._max_leverage

    @max_leverage.setter
    def max_leverage(self, max_leverage: float) -> None:
        self._max_leverage = max_leverage

        # if self.min_leverage == max_leverage:
        #     self.constraints[""]

        self.constraints["max_leverage"] = {
            "type": "ineq",
            "fun": lambda w: (max_leverage + 1.0) - np.sum(w),
        }

    @property
    def min_weight(self) -> float:
        return self._min_weight

    @min_weight.setter
    def min_weight(self, min_weight: float) -> None:
        self._min_weight = min_weight
        self.constraints["min_weight"] = {
            "type": "ineq",
            "fun": lambda w: w - min_weight,
        }

    @property
    def max_weight(self) -> Optional[float]:
        return self._max_weight

    @max_weight.setter
    def max_weight(self, max_weight: Optional[float]) -> None:
        if max_weight is None:
            return
        self._max_weight = max_weight
        self.constraints["max_weight"] = {
            "type": "ineq",
            "fun": lambda w: max_weight - w,
        }

    @property
    def min_return(self) -> Optional[float]:
        return self._min_return

    @min_return.setter
    def min_return(self, min_return: Optional[float]) -> None:
        if min_return is None:
            return
        if self.expected_returns is None:
            warnings.warn("unable to set minimum return constraint.")
            warnings.warn("expected returns is null.")
            return
        print(f"setting minimum return for opt {min_return}")
        minimum = self.expected_returns.min()
        maximum = self.expected_returns.max()
        min_return = min(maximum, max(min_return, minimum))
        self._min_return = min_return

        self.constraints["min_return"] = {
            "type": "ineq",
            "fun": lambda w: self.expected_return(w) - min_return,
        }

    @property
    def max_return(self) -> Optional[float]:
        return self._max_return

    @max_return.setter
    def max_return(self, max_return: Optional[float]) -> None:
        if max_return is None:
            return
        if self.expected_returns is None:
            warnings.warn("unable to set minimum return constraint.")
            warnings.warn("expected returns is null.")
            return
        minimum = self.expected_returns.min()
        maximum = self.expected_returns.max()
        max_return = min(maximum, max(max_return, minimum))
        self._max_return = max_return

        self.constraints["max_return"] = {
            "type": "ineq",
            "fun": lambda w: max_return - self.expected_return(w),
        }

    @property
    def min_volatility(self) -> Optional[float]:
        return self._min_volatility

    @min_volatility.setter
    def min_volatility(self, min_volatility: Optional[float]) -> None:
        if min_volatility is None:
            return
        if self.covariance_matrix is None:
            warnings.warn("unable to set minimum volatility constraint.")
            warnings.warn("covariance matrix is null.")
            return
        volatilities = np.sqrt(np.diag(self.covariance_matrix))
        minimum = min(volatilities)
        maximum = max(volatilities)
        min_volatility = min(maximum, max(min_volatility, minimum))
        self._min_volatility = min_volatility
        if min_volatility is None:
            return
        self.constraints["min_volatility"] = {
            "type": "ineq",
            "fun": lambda w: self.expected_volatility(w) - min_volatility,
        }

    @property
    def max_volatility(self) -> Optional[float]:
        return self._max_volatility

    @max_volatility.setter
    def max_volatility(self, max_volatility: Optional[float]) -> None:
        if max_volatility is None:
            return
        if self.covariance_matrix is None:
            warnings.warn("unable to set minimum volatility constraint.")
            warnings.warn("covariance matrix is null.")
            return
        volatilities = np.sqrt(np.diag(self.covariance_matrix))
        minimum = min(volatilities)
        maximum = max(volatilities)
        max_volatility = min(maximum, max(max_volatility, minimum))
        self._max_volatility = max_volatility
        if max_volatility is None:
            return
        self.constraints["max_volatility"] = {
            "type": "ineq",
            "fun": lambda w: max_volatility - self.expected_volatility(w),
        }

    @property
    def min_active_weight(self) -> Optional[float]:
        return self._min_active_weight

    @min_active_weight.setter
    def min_active_weight(self, min_active_weight: Optional[float]) -> None:
        if min_active_weight is None:
            return
        if self.weights_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            return

        self._min_active_weight = min_active_weight
        self.constraints["min_active_weight"] = {
            "type": "ineq",
            "fun": lambda w: np.sum(np.abs(w - np.array(self.weights_bm)))
            - min_active_weight,
        }

    @property
    def max_active_weight(self) -> Optional[float]:
        return self._max_active_weight

    @max_active_weight.setter
    def max_active_weight(self, max_active_weight: Optional[float]) -> None:
        if max_active_weight is None:
            return
        if self.weights_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            return
        self._max_active_weight = max_active_weight
        self.constraints["max_active_weight"] = {
            "type": "ineq",
            "fun": lambda w: max_active_weight
            - np.sum(np.abs(w - np.array(self.weights_bm))),
        }

    @property
    def min_exante_tracking_error(self) -> Optional[float]:
        return self._min_exante_tracking_error

    @min_exante_tracking_error.setter
    def min_exante_tracking_error(
        self, min_exante_tracking_error: Optional[float]
    ) -> None:
        """set maximum exante tracking error constraint"""
        if min_exante_tracking_error is None:
            return
        if self.weights_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            return
        self._min_exante_tracking_error = min_exante_tracking_error
        self.constraints["min_exante_tracking_error"] = {
            "type": "ineq",
            "fun": lambda w: self.exante_tracking_error(
                weights=w, weights_bm=np.array(self.weights_bm)
            )
            - min_exante_tracking_error,
        }

    @property
    def max_exante_tracking_error(self) -> Optional[float]:
        return self._max_exante_tracking_error

    @max_exante_tracking_error.setter
    def max_exante_tracking_error(
        self, max_exante_tracking_error: Optional[float]
    ) -> None:
        if max_exante_tracking_error is None:
            return
        """set maximum exante tracking error constraint"""
        if self.weights_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            return
        self._max_exante_tracking_error = max_exante_tracking_error
        self.constraints["max_exante_tracking_error"] = {
            "type": "ineq",
            "fun": lambda w: max_exante_tracking_error
            - self.exante_tracking_error(
                weights=w, weights_bm=np.array(self.weights_bm)
            ),
        }

    @property
    def min_expost_tracking_error(self) -> Optional[float]:
        return self._min_expost_tracking_error

    @min_expost_tracking_error.setter
    def min_expost_tracking_error(
        self, min_expost_tracking_error: Optional[float]
    ) -> None:
        if min_expost_tracking_error is None:
            return
        prices = self.prices
        prices_bm = self.prices_bm

        if prices is None or prices_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            raise ValueError("prices must not be none.")
        self._min_expost_tracking_error = min_expost_tracking_error
        self.constraints["min_expost_tracking_error"] = {
            "type": "ineq",
            "fun": lambda w: self.expost_tracking_error(
                weights=w,
                pri_returns_assets=np.array(prices.pct_change().fillna(0)),
                pri_returns_bm=np.array(prices_bm.pct_change().fillna(0)),
            )
            - min_expost_tracking_error,
        }

    @property
    def max_expost_tracking_error(self) -> Optional[float]:
        return self._max_expost_tracking_error

    @max_expost_tracking_error.setter
    def max_expost_tracking_error(
        self, max_expost_tracking_error: Optional[float]
    ) -> None:
        if max_expost_tracking_error is None:
            return
        prices = self.prices
        prices_bm = self.prices_bm

        if prices is None or prices_bm is None:
            warnings.warn("unable to set maximum active weight constraint.")
            warnings.warn("benchmark weights is null.")
            raise ValueError("prices must not be none.")

        self._max_expost_tracking_error = max_expost_tracking_error
        self.constraints["max_expost_tracking_error"] = {
            "type": "ineq",
            "fun": lambda w: max_expost_tracking_error
            - self.expost_tracking_error(
                weights=w,
                pri_returns_assets=np.array(prices.pct_change().fillna(0)),
                pri_returns_bm=np.array(prices_bm.pct_change().fillna(0)),
            ),
        }
