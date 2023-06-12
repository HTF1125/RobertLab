"""ROBERT"""
import pandas as pd
import streamlit as st
from pkg.src.core import data, factors


@st.cache_data()
def get_prices(*args, **kwargs) -> pd.DataFrame:
    """this is a pass through function"""
    return data.get_prices(*args, **kwargs)


@st.cache_data()
def get_factors(*args, **kwargs) -> pd.DataFrame:
    return factors.multi.multi_factor(tickers=kwargs.get("tickers"), features=args)


@st.cache_data()
def get_vix() -> pd.DataFrame:
    return data.get_prices(tickers="^VIX")

@st.cache_data()
def get_oecd_us_lei():
    return data.get_oecd_us_leading_indicator()
