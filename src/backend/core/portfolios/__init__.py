"""ROBERT"""
from .base import (
    EqualWeight,
    MaxReturn,
    MinCorrelation,
    MinVolatility,
    InverseVariance,
    HRP,
    HERC,
)

from .max_sharpe import MaxSharpe
from .risk_parity import RiskParity


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
