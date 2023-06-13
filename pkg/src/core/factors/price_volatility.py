"""ROBERT"""
from typing import List, Union, Set, Tuple
from pkg.src import data
from .base import Factors


__all__ = [
    "PriceVolatility1M",
    "PriceVolatility3M",
]


class PriceVolatility(Factors):
    months = 1

    def __init__(self, tickers: Union[str, List, Set, Tuple]) -> None:
        super().__init__(tickers=tickers)
        self.factors = (
            data.get_prices(tickers=self.tickers)
            .pct_change()
            .rolling(21 * self.months)
            .std()
        )


class PriceVolatility1M(PriceVolatility):
    months = 1


class PriceVolatility3M(PriceVolatility):
    months = 3
