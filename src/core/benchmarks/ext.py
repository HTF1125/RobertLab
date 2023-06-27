"""ROBERT"""
import pandas as pd
from src.core import universes
from .base import Benchmark


class UnitedStates64(Benchmark):
    UNIVERSE = universes.UsAllocation()

    def calculate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = prices.copy().dropna()
        w["SPY"] = 0.6
        w["AGG"] = 0.4
        return w


class Global64(Benchmark):
    UNIVERSE = universes.GlobalAllocation()

    def calculate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = prices.copy().dropna()
        w["ACWI"] = 0.6
        w["BND"] = 0.4
        return w


class UnitedStates55(Benchmark):
    UNIVERSE = universes.UsAllocation()

    def calculate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = prices.copy().dropna()
        w["SPY"] = 0.5
        w["AGG"] = 0.5
        return w


class EqualWeightBenchmark(Benchmark):
    def calculate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = prices.dropna(how="all").copy()
        w = 1 / p.count(axis=1)
        w = p.mask(p.notnull(), w, axis=0)
        return w


class UnitedStatesSectorsEW(EqualWeightBenchmark):
    UNIVERSE = universes.UnitedStatesSectors()


class GlobalAssetAllocationEW(EqualWeightBenchmark):
    UNIVERSE = universes.GlobalAssetAllocation()

