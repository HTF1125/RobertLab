"""ROBERT"""
from typing import Union
from typing import overload
import pandas as pd


@overload
def to_momentum(prices: pd.Series, months: int = 1, skip_months: int = 0) -> pd.Series:
    ...


@overload
def to_momentum(
    prices: pd.DataFrame, months: int = 1, skip_months: int = 0
) -> pd.DataFrame:
    ...


def to_momentum(
    prices: Union[pd.DataFrame, pd.Series], months: int = 1, skip_months: int = 0
) -> Union[pd.DataFrame, pd.Series]:
    return prices.pct_change(periods=(months - skip_months) * 21).shift(
        skip_months * 21
    )
