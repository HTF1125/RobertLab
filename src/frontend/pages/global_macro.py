"""ROBERT"""
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.core import metrics
from .base import BasePage
from .. import data
import yfinance as yf
import pandas_datareader as pdr


@st.cache_data(ttl=pd.Timedelta(days=1))
def get_dollar_yoy_index():
    price = yf.download(tickers="DX-Y.NYB", progress=False)["Adj Close"]
    price.name = "Dollar Index"
    return price.pct_change(252).dropna()


@st.cache_data(ttl=pd.Timedelta(days=1))
def get_spy_yoy_10yr_ma():
    price = yf.download(tickers="^GSPC", progress=False)["Adj Close"]
    price.name = "S&P500"
    return price.pct_change(252).rolling(252 * 10).mean().dropna()


@st.cache_data(ttl=pd.Timedelta(days=1))
def get_debt_to_gdp_10yoy():
    price = pdr.DataReader("GFDEGDQ188S", "fred", "1960-1-1")
    return price.pct_change(4 * 10).dropna()


@st.cache_data(ttl=pd.Timedelta(days=1))
def get_copper_yoy():
    price = yf.download("HG=F, CL=F", progress=False)["Adj Close"]
    return price.pct_change(252)


@st.cache_data(ttl=pd.Timedelta(days=1))
def get_us_cpi():
    cpi = (
        pdr.DataReader("CPIAUCSL", "fred", "1960-1-1")
        .resample("M")
        .last()
        .pct_change(12)
    )
    inf_exp_10y = (
        pdr.DataReader("T10YIE", "fred", "1960-1-1").resample("M").last() / 100
    )
    cpi_ma = cpi.rolling(10 * 12).mean()
    cpi_ma.columns = ["Cpi Ma"]
    return pd.concat([cpi, cpi_ma, inf_exp_10y], axis=1)


@st.cache_data(ttl=pd.Timedelta(days=1))
def get_us_pop():
    price = pdr.DataReader("POPTHM", "fred", "1960-1-1")
    return price.iloc[:, 0].pct_change(12).dropna()


def plot_spy_yoy_10yr_ma():
    spy_yoy_10yr_ma = get_spy_yoy_10yr_ma()
    st.plotly_chart(
        go.Figure()
        .add_trace(go.Scatter(x=spy_yoy_10yr_ma.index, y=spy_yoy_10yr_ma.values))
        .update_layout(
            hovermode="x unified",
            title="S&P500 YoY% 10Yr Moving Average",
            xaxis_tickformat="%Y-%m-%d",
            yaxis_tickformat=".2%",
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )


def plot_dollar_yoy_index():
    dollar_yoy_index = get_dollar_yoy_index()

    st.plotly_chart(
        go.Figure()
        .add_trace(go.Scatter(x=dollar_yoy_index.index, y=dollar_yoy_index.values))
        .update_layout(
            hovermode="x unified",
            title="US Dollar Index YoY%",
            xaxis_tickformat="%Y-%m-%d",
            yaxis_tickformat=".2%",
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )


def get_us_2y_6mom():
    us_2y = data.get_us_2y_yield()
    us_2y_6mom = us_2y.pct_change(121).dropna()
    st.plotly_chart(
        go.Figure()
        .add_trace(
            go.Scatter(
                x=us_2y_6mom.index,
                y=us_2y_6mom.values,
            )
        )
        .update_layout(
            hovermode="x unified",
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )


def get_vix_regime():
    vix = data.get_vix()
    normalized = metrics.rolling.to_standard_scaler(vix, window=252).clip(
        lower=-3, upper=3
    )
    normalized = normalized.ewm(90).mean().dropna()
    st.plotly_chart(
        go.Figure()
        .add_trace(
            go.Scatter(x=normalized.index, y=normalized.values, hovertemplate="")
        )
        .update_layout(
            hovermode="x unified",
            xaxis_tickformat="%Y-%m-%d",
            yaxis_tickformat=".2%",
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )


class GlobalMacro(BasePage):
    def load_page(self):
        plot_spy_yoy_10yr_ma()
        plot_dollar_yoy_index()
        get_vix_regime()
        get_us_2y_6mom()

    def get_oecd_us_lei_regime(self, start, end):
        lei = data.get_oecd_us_lei()
        lei.index = lei.index + pd.DateOffset(months=1)
        change = lei.resample("M").last().diff().dropna()
        normalized = metrics.rolling.to_standard_scaler(change, window=12 * 5)
        normalized = normalized.clip(lower=-3, upper=3)
        st.plotly_chart(
            self.bar(data=normalized.dropna().loc[start:end]),
            use_container_width=True,
        )

    def inflation_short_yield_data(self, start=None, end=None) -> pd.DataFrame:
        """
        Get raw data for leading economic indicator regime.
        """
        tickers = dict(THREEFFTP10="term", T10YIE="inflation", DGS10="treasury")

        start = start if start else "1900-01-01"
        import pandas_datareader as pdr

        dt = pdr.DataReader(list(tickers.keys()), "fred", start=start)
        dt = dt.rename(columns=tickers)
        dt["short_yield"] = dt["treasury"] - dt["term"] - dt["inflation"]
        dt = dt[["short_yield", "inflation"]]
        return dt.dropna()
