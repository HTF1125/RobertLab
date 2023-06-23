"""ROBERT"""
import os
import json
import pandas as pd
import streamlit as st
from pkg.src import data


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

@st.cache_data()
def get_universe():
    file = os.path.join(os.path.dirname(__file__), "universe.json")
    with open(file=file, mode="r", encoding="utf-8") as json_file:
        return json.load(json_file)
