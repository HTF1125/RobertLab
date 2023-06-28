"""ROBERT"""
import sys
from .base import Benchmark
from .ext import (
    UnitedStates55,
    UnitedStates64,
    Global64,
    GlobalAssetAllocationEW,
    UnitedStatesSectorsEW,
)

__all__ = [
    "UnitedStatesSectorsEW",
    "UnitedStates55",
    "UnitedStates64",
    "Global64",
    "GlobalAssetAllocationEW",
]


def get(benchmark: str) -> Benchmark:
    # Use getattr() to get the attribute value
    try:
        return getattr(sys.modules[__name__], benchmark)()
    except AttributeError as exc:
        raise ValueError(f"Invalid factor: {benchmark}") from exc