"""ROBERT"""
import sys
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


def get(optimizer: str) -> "Portfolio":
    # Use getattr() to get the attribute value
    try:
        return getattr(sys.modules[__name__], optimizer)()
    except AttributeError as exc:
        raise ValueError(f"Invalid optimizer: {optimizer}") from exc
