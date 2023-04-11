import pandas as pd
from .metrics import vectorize

def moving_average(prices: pd.DataFrame, window: int = 5) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        window (int, optional): _description_. Defaults to 5.

    Returns:
        pd.Series: _description_
    """
    return prices.ffill().iloc[-window:].mean().dropna()


def momentum(prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    idx = prices.index.copy()
    resampled_prices = prices.resample("D").last().ffill()
    offset_prices = resampled_prices.shift(1, freq=pd.DateOffset(**kwargs))
    return (prices / offset_prices).loc[idx]


@vectorize
def moving_average(prices: pd.Series, window: int = 20) -> pd.Series:
    """_summary_

    Args:
        prices (pd.Series): _description_
        window (int, optional): _description_. Defaults to 20.

    Returns:
        pd.Series: _description_
    """
    return prices.rolling(window=window).mean()


