"""ROBERT"""


import streamlit as st
from plotly.graph_objects import Scatter
from plotly.subplots import make_subplots
from pkg.src.core import data

def make_yield_curve():
    yield_curve = {"T10Y2Y": "10Y-2Y", "T10Y3M": "10Y-3M"}

    yield_curve_data = data.get_macro(list(yield_curve.keys())).loc["2019":]
    yield_curve_data = yield_curve_data.rename(columns=yield_curve)

    fig = make_subplots(rows=1, cols=1)

    for x in yield_curve_data:

        price_trace = Scatter(
            x=yield_curve_data.index,
            y=yield_curve_data[x],
            name=x,
            hovertemplate="Date: %{x} %{y}",
        )
        fig.add_trace(price_trace)

    fig.update_layout(
        title="Performance",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x",
        legend=dict(orientation="v", yanchor="top", y=1.1, xanchor="left", x=0),
    )

    fig.update_layout(
        xaxis=dict(
            title="Date", tickformat="%Y-%m-%d"
        ),  # Customize the date format
        yaxis=dict(
            title="Price",
            tickprefix="", # Add a currency symbol to the y-axis tick labels
            ticksuffix="%"
        ),
    )

    st.plotly_chart(fig, use_container_width=True)



def make_yield_curve2():
    yield_curve = {
        "DGS1MO": "1M",
        "DGS3MO": "3M",
        "DGS6MO": "6M",
        "DGS1" : "1Y",
        "DGS2" : "2Y",
        "DGS3" : "3Y",
        "DGS5" : "5Y",
        "DGS7" : "7Y",
        "DGS10" : "10Y",
        "DGS20" : "20Y",
        "DGS30" : "30Y",
    }

    yield_curve_data = data.get_macro(list(yield_curve.keys()))
    yield_curve_data = yield_curve_data.rename(columns=yield_curve).loc["2019":]

    fig = make_subplots(rows=1, cols=1)

    for x in yield_curve_data:

        price_trace = Scatter(
            x=yield_curve_data.index,
            y=yield_curve_data[x],
            name=x,
            hovertemplate="Date: %{x} %{y}",
        )
        fig.add_trace(price_trace)

    fig.update_layout(
        title="Performance",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x",
        legend=dict(orientation="v", yanchor="top", y=1.1, xanchor="left", x=0),
    )

    fig.update_layout(
        xaxis=dict(
            title="Date", tickformat="%Y-%m-%d"
        ),  # Customize the date format
        yaxis=dict(
            title="Price",
            tickprefix="", # Add a currency symbol to the y-axis tick labels
            ticksuffix="%"
        ),
    )

    st.plotly_chart(fig, use_container_width=True)




def make_inflation_linked():
    yield_curve = {
        "DFII5" : "5Y",
        "DFII7" : "7Y",
        "DFII10" : "10Y",
        "DFII20" : "20Y",
        "DFII30" : "30Y",
    }

    yield_curve_data = data.get_macro(list(yield_curve.keys()))
    yield_curve_data = yield_curve_data.rename(columns=yield_curve).loc["2019":]

    fig = make_subplots(rows=1, cols=1)

    for x in yield_curve_data:

        price_trace = Scatter(
            x=yield_curve_data.index,
            y=yield_curve_data[x],
            name=x,
            hovertemplate="Date: %{x} %{y}",
        )
        fig.add_trace(price_trace)

    fig.update_layout(
        title="Performance",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x",
        legend=dict(orientation="v", yanchor="top", y=1.1, xanchor="left", x=0),
    )

    fig.update_layout(
        xaxis=dict(
            title="Date", tickformat="%Y-%m-%d"
        ),  # Customize the date format
        yaxis=dict(
            title="Price",
            tickprefix="", # Add a currency symbol to the y-axis tick labels
            ticksuffix="%"
        ),
    )

    st.plotly_chart(fig, use_container_width=True)