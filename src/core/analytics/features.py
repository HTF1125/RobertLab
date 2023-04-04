import pandas as pd


def moving_average(prices: pd.DataFrame, window: int = 5) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        window (int, optional): _description_. Defaults to 5.

    Returns:
        pd.Series: _description_
    """
    return prices.ffill().iloc[-window:].mean().dropna()


def momentum(prices: pd.DataFrame, years: int = 1) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        years (int, optional): _description_. Defaults to 1.

    Returns:
        pd.Series: _description_
    """
    pr_date = pd.to_datetime(str(prices.index[-1])) - pd.DateOffset(years=years)

    return prices.iloc[-1] / prices.iloc[prices.index.get_loc(pr_date)] - 1