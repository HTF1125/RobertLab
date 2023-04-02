import pandas as pd


def moving_average(prices: pd.DataFrame, window: int = 5) -> pd.Series:

    return prices.ffill().iloc[-window:].mean().dropna()


def momentum(prices: pd.DataFrame, years: int = 1) -> pd.Series:

    pr_date = prices.index[-1] - pd.DateOffset(years = 1)

    return prices.iloc[-1] / prices.iloc[prices.index.get_loc(pr_date)] - 1