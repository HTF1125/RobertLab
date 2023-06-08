"""ROBERT"""
from typing import Union, List, Set, Tuple, Callable
import pandas as pd
from pkg.src.core import metrics
from . import base

def multi_factor(
    tickers: Union[str, List, Set, Tuple],
    features: List[Callable],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    factors = []
    for feature in features:
        func = feature if callable(feature) else getattr(base, feature)
        factors.append(func(tickers=tickers, normalize=normalize))
    return metrics.to_multi_factor(factors=factors)
