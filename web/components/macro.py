"""ROBERT"""


import streamlit as st
from pkg.src.core import data
from .charts import line


def make_yield_curve():
    with st.spinner(text="Loading data..."):
        yield_curve = {"T10Y2Y": "10Y-2Y", "T10Y3M": "10Y-3M"}

        yield_curve_data = data.get_macro(list(yield_curve.keys())).loc["2019":]
        yield_curve_data = yield_curve_data.rename(columns=yield_curve)
        fig = line(yield_curve_data)

        st.plotly_chart(fig, use_container_width=True)


def make_yield_curve2():
    yield_curve = {
        "DGS1MO": "1M",
        "DGS3MO": "3M",
        "DGS6MO": "6M",
        "DGS1": "1Y",
        "DGS2": "2Y",
        "DGS3": "3Y",
        "DGS5": "5Y",
        "DGS7": "7Y",
        "DGS10": "10Y",
        "DGS20": "20Y",
        "DGS30": "30Y",
    }
    with st.spinner(text="Loading data..."):

        yield_curve_data = data.get_macro(list(yield_curve.keys()))
        yield_curve_data = yield_curve_data.rename(columns=yield_curve).loc["2019":]
        fig = line(yield_curve_data)

        st.plotly_chart(fig, use_container_width=True)


def make_inflation_linked():
    yield_curve = {
        "DFII5": "5Y",
        "DFII7": "7Y",
        "DFII10": "10Y",
        "DFII20": "20Y",
        "DFII30": "30Y",
    }
    with st.spinner(text="Loading data..."):

        yield_curve_data = data.get_macro(list(yield_curve.keys()))
        yield_curve_data = yield_curve_data.rename(columns=yield_curve).loc["2019":]
        fig = line(yield_curve_data)
        st.plotly_chart(fig, use_container_width=True)
