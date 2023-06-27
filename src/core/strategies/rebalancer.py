"""ROBERT"""
from typing import Optional, List, Dict, Any, Union
import pandas as pd
from src.core.factors import MultiFactors
from src.core.portfolios import PortfolioOptimizer
from src.core.strategies import Strategy

from .parser import Parser


class Rebalancer:
    def __init__(
        self,
        optimizer: Union[str, PortfolioOptimizer] = "EqualWeight",
        factors: Optional[MultiFactors] = None,
        optimizer_constraints: Optional[Dict[str, float]] = None,
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
        span: Optional[int] = None,
        risk_free: float = 0.0,
        case: bool = False,
    ) -> None:
        self.optimizer = Parser.get_optimizer(optimizer)
        self.optimizer_constraints = optimizer_constraints or {}
        self.specific_constraints = specific_constraints or []
        self.factors = factors
        self.span = span
        self.risk_free = risk_free

        if not case:
            if self.factors:
                self.factors_data = self.factors.standard_scaler()
            else:
                self.factors_data = None
        else:
            self.factors_data = None

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
            **kwargs,
            **self.optimizer_constraints,
        ).set_specific_constraints(self.specific_constraints)

        return opt.solve()

    def get_signature(self) -> Dict:
        return {
            "optimizer": self.optimizer.__class__.__name__,
            "factors": self.factors.factors
            if isinstance(self.factors, MultiFactors)
            else None,
            "optimizer_constraints": self.optimizer_constraints,
            "specific_constraints": self.specific_constraints,
        }
