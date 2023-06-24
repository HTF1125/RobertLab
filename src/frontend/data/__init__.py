"""ROBERT"""
import os
import json
from typing import Union, List, Set, Tuple
import pandas as pd
import yfinance as yf
import streamlit as st
from src.backend import data


def get_prices(tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
    # create ticker list
    tickers = (
        tickers
        if isinstance(tickers, (list, set, tuple))
        else tickers.replace(",", " ").split()
    )
    out = []
    for ticker in tickers:
        price = get_price(ticker)
        if price is not None:
            out.append(price)
    return pd.concat(out, axis=1).sort_index()


@st.cache_data(ttl="1d")
def get_price(ticker: str) -> pd.Series:
    p = yf.download(ticker, progress=False)["Adj Close"]
    p.name = ticker
    return p


@st.cache_data(ttl="1d")
def get_vix() -> pd.Series:
    return data.get_price(ticker="^VIX")


def get_us_2y_yield() -> pd.Series:
    return data.get_macro(tickers="DGS2").iloc[:, 0]


@st.cache_data(ttl="1d")
def get_oecd_us_lei():
    return data.get_oecd_us_leading_indicator()


@st.cache_data(ttl="1d")
def get_universe():
    file = os.path.join(os.path.dirname(__file__), "universe.json")
    with open(file=file, mode="r", encoding="utf-8") as json_file:
        return json.load(json_file)
