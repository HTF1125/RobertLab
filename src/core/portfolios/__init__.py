"""ROBERT"""
import sys
from typing import Union, Type
from .base import Portfolio
from .ext import *

__all__ = [
    "EqualWeight",
    "MaxReturn",
    "MaxSharpe",
    "MinVolatility",
    "MinCorrelation",
    "InverseVariance",
    "RiskParity",
    "HRP",
    "HERC",
]


def get(portfolio: Union[str, Portfolio, Type[Portfolio]]) -> Portfolio:
    # Use getattr() to get the attribute value

    try:
        if isinstance(portfolio, str):
            return getattr(sys.modules[__name__], portfolio)()
        if isinstance(portfolio, type) and issubclass(portfolio, Portfolio):
            return portfolio()
        if issubclass(portfolio.__class__, Portfolio):
            return portfolio
        return getattr(sys.modules[__name__], str(portfolio))()
    except AttributeError as exc:
        raise ValueError(f"Invalid portfolio: {portfolio}") from exc
