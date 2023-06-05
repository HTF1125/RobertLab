"""ROBERT"""
from typing import List, Union, Set, Tuple, Callable
from functools import lru_cache
import pandas as pd
from .. import data
from .. import metrics


def price_momentum(
    tickers: Union[str, List, Set, Tuple],
    months: int = 1,
    skip_months: int = 0,
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    prices = data.get_prices(tickers=tickers)
    momentum = metrics.rolling.to_momentum(
        prices=prices, months=months, skip_months=skip_months
    )
    if normalize == "standard_percentile":
        return momentum.apply(metrics.to_standard_percentile, axis=1)
    return momentum.apply(metrics.to_standard_scalar, axis=1)


def price_momentum_1m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(tickers=tickers, months=1, skip_months=0, normalize=normalize)


def price_momentum_3m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(tickers=tickers, months=3, skip_months=0, normalize=normalize)


def price_momentum_6m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(tickers=tickers, months=6, skip_months=0, normalize=normalize)


def price_momentum_6m_1m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(tickers=tickers, months=6, skip_months=1, normalize=normalize)


def price_momentum_6m_2m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(tickers=tickers, months=6, skip_months=2, normalize=normalize)


def price_momentum_12m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(
        tickers=tickers, months=12, skip_months=0, normalize=normalize
    )


def price_momentum_12m_1m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(
        tickers=tickers, months=12, skip_months=1, normalize=normalize
    )


def price_momentum_12m_2m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(
        tickers=tickers, months=12, skip_months=2, normalize=normalize
    )


def price_momentum_24m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(
        tickers=tickers, months=24, skip_months=0, normalize=normalize
    )


def price_momentum_36m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(
        tickers=tickers, months=36, skip_months=0, normalize=normalize
    )


def price_momentum_24m_1m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(
        tickers=tickers, months=24, skip_months=1, normalize=normalize
    )


def price_momentum_36m_1m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(
        tickers=tickers, months=36, skip_months=1, normalize=normalize
    )


def price_momentum_24m_2m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(
        tickers=tickers, months=24, skip_months=2, normalize=normalize
    )


def price_momentum_36m_2m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(
        tickers=tickers, months=36, skip_months=2, normalize=normalize
    )


__all__ = [
    "price_momentum_1m",
    "price_momentum_3m",
    "price_momentum_6m",
    "price_momentum_6m_1m",
    "price_momentum_6m_2m",
    "price_momentum_12m",
    "price_momentum_12m_1m",
    "price_momentum_12m_2m",
    "price_momentum_24m",
    "price_momentum_24m_1m",
    "price_momentum_24m_2m",
    "price_momentum_36m",
    "price_momentum_36m_1m",
    "price_momentum_36m_2m",
]
