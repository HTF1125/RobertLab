"""ROBERT"""
from typing import Optional, List, Dict, Any, Union
import pandas as pd
from src.core import portfolios
from src.core.factors import MultiFactors
from src.core.strategies import Strategy

from .parser import Parser


class Rebalancer:
    def __init__(
        self,
        optimizer: Union[str, portfolios.Optimizer] = "EqualWeight",
        factors: MultiFactors = MultiFactors(),
        optimizer_constraints: Optional[Dict[str, float]] = None,
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
        span: Optional[int] = None,
        risk_free: float = 0.0,
    ) -> None:
        self.optimizer = Parser.get_optimizer(optimizer)
        self.optimizer_constraints = optimizer_constraints or {}
        self.specific_constraints = specific_constraints or []
        self.factors = factors
        self.span = span
        self.risk_free = risk_free

    def __call__(self, strategy: Strategy) -> pd.Series:
        """Calculate portfolio allocation weights based on the Strategy instance.

        Args:
            strategy (Strategy): Strategy instance.

        Returns:
            pd.Series: portfolio allocation weights.
        """
        kwargs = (
            {
                "prices_bm": strategy.benchmark.get_performance(strategy.date),
                "weights_bm": strategy.benchmark.get_weights(strategy.date),
            }
            if strategy.benchmark is not None
            else {}
        )
        prices = strategy.reb_prices
        factors = (
            self.factors.get_factor_by_date(date=strategy.date)
            if self.factors
            else None
        )

        opt = self.optimizer.from_prices(
            prices=prices,
            factors=factors,
            span=self.span,
            risk_free=self.risk_free,
            **kwargs,
            **self.optimizer_constraints,
        ).set_specific_constraints(self.specific_constraints)

        return opt.solve()

    def get_signature(self) -> Dict:
        return {
            "optimizer": self.optimizer.__class__.__name__,
            "factors": list(self.factors.keys()),
            "optimizer_constraints": self.optimizer_constraints,
            "specific_constraints": self.specific_constraints,
        }
