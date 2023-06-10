"""ROBERT"""

import pandas as pd
import streamlit as st
from pkg.src.core import data
from ..components import charts
import plotly.graph_objects as go


def to_macd(
    prices: pd.DataFrame,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> pd.DataFrame:
    MACD = (
        +prices.ewm(span=fast_window, min_periods=fast_window).mean()
        - prices.ewm(span=slow_window, min_periods=slow_window).mean()
    )
    signal = MACD.ewm(span=signal_window, min_periods=slow_window).mean()
    return signal


@st.cache_data()
def get_vix() -> pd.DataFrame:
    return data.get_prices(tickers="^VIX")


def main():
    mean = get_vix().rolling(252).mean()
    std = get_vix().rolling(252).std()
    result = (get_vix() - mean) / std
    result = result.ewm(90).mean()
    result = result.clip(-3, 3)
    # Display the plot
    st.plotly_chart(
        charts.bar(data=result),
        use_container_width=True,
    )
    df = data.get_oecd_us_leading_indicator()
    df.index = df.index + pd.DateOffset(months=1)
    df = df.resample("M").last().dropna()

    df = df.diff()
    mean = df.rolling(12 * 5).mean()
    std = df.rolling(12 * 5).std()
    normalized = (df - mean) / std

    normalized = normalized.clip(-3, 3).dropna()

    # Display the plot
    st.plotly_chart(
        charts.bar(data=normalized),
        use_container_width=True,
    )