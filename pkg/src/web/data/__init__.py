"""ROBERT"""
import pandas as pd
import streamlit as st
from pkg.src import data
from pkg.src.core import factors


@st.cache_data()
def get_prices(*args, **kwargs) -> pd.DataFrame:
    """this is a pass through function"""
    return data.get_prices(*args, **kwargs)


# @st.cache_data()
# def get_factors(factor_list) -> pd.DataFrame:
#     return factors.multi.MultiFactors(
#         tickers=kwargs.get("tickers"), factor_list=factor_list
#     )


@st.cache_data()
def get_vix() -> pd.DataFrame:
    return data.get_prices(tickers="^VIX")


@st.cache_data()
def get_oecd_us_lei():
    return data.get_oecd_us_leading_indicator()
