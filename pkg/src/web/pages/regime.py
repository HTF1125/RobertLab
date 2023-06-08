"""ROBERT"""

import numpy as np
import pandas as pd
import streamlit as st
from pkg.src.core import data, metrics
from ..components import charts

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
    st.plotly_chart(charts.line(result), use_container_width=True)

