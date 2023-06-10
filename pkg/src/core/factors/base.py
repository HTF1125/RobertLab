"""ROBERT"""
from typing import List, Union, Set, Tuple
import numpy as np
import pandas as pd
from .. import data
from .. import metrics


def price_momentum(
    tickers: Union[str, List, Set, Tuple],
    months: int = 1,
    skip_months: int = 0,
    absolute: bool = False,
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    prices = data.get_prices(tickers=tickers)
    momentum = metrics.rolling.to_momentum(
        prices=prices, months=months, skip_months=skip_months, absolute=absolute
    )
    if normalize == "standard_percentile":
        return momentum.apply(metrics.to_standard_percentile, axis=1)
    return momentum.apply(metrics.to_standard_percentile, axis=1)


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


def price_momentum_9m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(tickers=tickers, months=9, skip_months=0, normalize=normalize)


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


def price_momentum_18m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(
        tickers=tickers, months=18, skip_months=0, normalize=normalize
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


def absolute_price_momentum_12m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    return price_momentum(
        tickers=tickers, months=12, skip_months=0, absolute=True, normalize=normalize
    )


def volatility_3m(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    prices = data.get_prices(tickers=tickers)
    volatility = metrics.to_pri_return(prices=prices).rolling(21 * 3).std()
    if normalize == "standard_percentile":
        return volatility.apply(metrics.to_standard_percentile, axis=1)
    return volatility.apply(metrics.to_minmax_scalar, axis=1)


def price_momentum_diffusion(
    tickers: Union[str, List, Set, Tuple],
    normalize: str = "standard_percentile",
) -> pd.DataFrame:
    """
    Momentum Diffusion summarizes the net price momentum at different frequencies
    and is constructed to fluctuate between +1 and -1.
    When the index is +1, it implies that upward price momentum is persistent.
    The various frequencies used are 1M, 2M, 3M, 6M, 9M, 12M. Stocks wit higher
    persistence in momentum are allocated to the top portfolio.
    """
    prices = data.get_prices(tickers=tickers)
    momentums = (
        pd.concat(
            objs=[
                metrics.rolling.to_momentum(prices=prices, months=1).stack(),
                metrics.rolling.to_momentum(prices=prices, months=2).stack(),
                metrics.rolling.to_momentum(prices=prices, months=3).stack(),
                metrics.rolling.to_momentum(prices=prices, months=6).stack(),
                metrics.rolling.to_momentum(prices=prices, months=9).stack(),
                metrics.rolling.to_momentum(prices=prices, months=12).stack(),
            ],
            axis=1,
        )
        .apply(np.sign)
        .sum(axis=1)
        .unstack()
        .apply(np.sign)
    )
    return momentums


__all__ = [
    "absolute_price_momentum_12m",
    "price_momentum_1m",
    "price_momentum_3m",
    "price_momentum_6m",
    "price_momentum_6m_1m",
    "price_momentum_6m_2m",
    "price_momentum_9m",
    "price_momentum_12m",
    "price_momentum_12m_1m",
    "price_momentum_12m_2m",
    "price_momentum_18m",
    "price_momentum_24m",
    "price_momentum_24m_1m",
    "price_momentum_24m_2m",
    "price_momentum_36m",
    "price_momentum_36m_1m",
    "price_momentum_36m_2m",
    "price_momentum_diffusion",
    "volatility_3m",
]
