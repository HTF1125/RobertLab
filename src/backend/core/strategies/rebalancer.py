"""ROBERT"""
from typing import Optional, List, Dict, Any, Type
import pandas as pd
from src.backend.core import portfolios, strategies
from src.backend.core.factors import MultiFactors
from src.backend.core.portfolios.base import PortfolioOptimizer


class Rebalancer:
    def __init__(
        self,
        optimizer: str = "EqualWeight",
        factors: Optional[MultiFactors] = None,
        optimizer_constraints: Optional[Dict[str, float]] = None,
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
        span: Optional[int] = None,
        risk_free: float = 0.0,
        prices_bm: Optional[pd.Series] = None,
        weights_bm: Optional[pd.Series] = None,
    ) -> None:
        super().__init__()
        self.optimizer: Type[PortfolioOptimizer] = getattr(portfolios, optimizer)
        self.optimizer_constraints = optimizer_constraints or {}
        self.specific_constraints = specific_constraints or []
        self.factors = factors
        self.span = span
        self.risk_free = risk_free
        self.prices_bm = prices_bm
        self.weights_bm = weights_bm

        if self.factors:
            self.factors_data = self.factors.standard_scaler
        else:
            self.factors_data = None

    def __call__(self, strategy: strategies.Strategy) -> pd.Series:
        """Calculate portfolio allocation weights based on the Strategy instance.

        Args:
            strategy (Strategy): Strategy instance.

        Returns:
            pd.Series: portfolio allocation weights.
        """
        prices = strategy.reb_prices
        opt = self.optimizer.from_prices(
            prices=prices,
            factors=self.factors_data.loc[: strategy.date]
            .iloc[-1]
            .reindex(prices.columns, fill_value=0.0)
            .fillna(0)
            if isinstance(self.factors_data, pd.DataFrame)
            else None,
            span=self.span,
            risk_free=self.risk_free,
            prices_bm=self.prices_bm,
            weights_bm=self.weights_bm,
            **self.optimizer_constraints,
        ).set_specific_constraints(self.specific_constraints)

        return opt.solve()

    def get_signiture(self) -> Dict:
        return {
            "optimizer": self.optimizer.__name__,
            "factors": self.factors.factors
            if isinstance(self.factors, MultiFactors)
            else None,
            "optimizer_constraints": self.optimizer_constraints,
            "specific_constraints": self.specific_constraints,
        }
