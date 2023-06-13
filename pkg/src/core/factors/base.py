"""ROBERT"""
from typing import List, Union, Set, Tuple, Dict
import numpy as np
import pandas as pd
from .. import metrics
from ... import data


# The Factors class contains a method that returns a DataFrame of standard
# percentiles for the factors.
class Factors(object):
    factors = pd.DataFrame()

    def __init__(self, tickers: Union[str, List, Set, Tuple]) -> None:
        self.tickers = tickers

    def __repr__(self) -> str:
        return "Base Factors"

    def __str__(self) -> str:
        return "Base Factors"

    @property
    def standard_percentile(self) -> pd.DataFrame:
        return self.factors.apply(metrics.to_standard_percentile, axis=1)
