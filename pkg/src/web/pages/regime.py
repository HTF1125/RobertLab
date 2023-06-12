"""ROBERT"""
from datetime import datetime
import pandas as pd
import streamlit as st
from ..components import charts
from .. import data
from ...core import metrics

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


def get_vix_regime(start, end):

    vix = data.get_vix()
    normalized = metrics.rolling.to_standard_scalar(vix, window=252)
    normalized = normalized.ewm(90).mean()
    normalized = normalized.clip(lower=-3, upper=3)
    st.plotly_chart(
        charts.bar(data=normalized.loc[start:end]),
        use_container_width=True,
    )

def get_oecd_us_lei_regime(start, end):

    lei = data.get_oecd_us_lei()
    lei.index = lei.index + pd.DateOffset(months=1)
    change = lei.resample("M").last().diff().dropna()
    normalized = metrics.rolling.to_standard_scalar(change, window=12 * 5)
    normalized = normalized.clip(lower=-3, upper=3)
    st.plotly_chart(
        charts.bar(data=normalized.dropna().loc[start:end]),
        use_container_width=True,
    )



def main():
    dates = pd.date_range("1990-1-1", datetime.now(), freq="D")
    start, end = st.select_slider(
        label="Select Date Range",
        options=dates,
        value=(dates[0], dates[-1]),
        format_func=lambda x: f"{x:%Y-%m-%d}"
    )


    get_vix_regime(start=start, end=end)
    get_oecd_us_lei_regime(start=start, end=end)
