

import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(path) != "src":
    path = os.path.abspath(os.path.join(path, "../"))
    break
sys.path.append(path)
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from core.strategies import BacktestManager
from core import data
from core import metrics
from web import components

st.set_page_config(
    page_title="ROBERT'S WEBSITE",
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f440.png",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

			html, body, [class*="css"]  {
			font-family: 'Roboto', sans-serif;
			}
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)


hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


def get_backtestmanager() -> BacktestManager:
    # Initialization
    if "backtestmanager" not in st.session_state:
        st.session_state["backtestmanager"] = BacktestManager()
    return st.session_state["backtestmanager"]


st.session_state["universe"] = st.selectbox(
    label="Select Investment Universe",
    options=["USSECTORETF", "General"],
)


get_backtestmanager().set_universe(name=st.session_state["universe"])


def clear_strategies():
    get_backtestmanager().reset_strategies()


momentum_tab, base_tab = st.tabs(["Momentum", "Base"])

with momentum_tab:
    st.button(label="Clear Strategies", on_click=clear_strategies)

    with st.form(key="momentum_month"):

        (
            objective,
            start,
            end,
            frequency,
            commission,
        ) = components.get_strategy_general_params()

        cols = st.columns([1] * 3)

        months = cols[0].select_slider(
            label="Momentum Months", options=range(1, 36 + 1), value=1
        )
        skip_months = cols[1].select_slider(
            label="Momentum Skip Months", options=range(0, 6 + 1), value=0
        )

        target_percentile = cols[2].select_slider(
            label="Target Percentile",
            options=range(0, 100 + 10, 10),
            value=70,
        )

        absolute = st.checkbox(label="Absolute Momentum", value=False)

        submitted = st.form_submit_button("Submit")
        if submitted:
            get_backtestmanager().commission = int(commission)
            get_backtestmanager().start = str(start)
            get_backtestmanager().end = str(end)
            get_backtestmanager().frequency = frequency
            with st.spinner(text="Backtesting in progress..."):

                get_backtestmanager().Momentum(
                    months=months,
                    skip_months=skip_months,
                    objective=objective,
                    absolute=absolute,
                    target_percentile=target_percentile / 100,
                )



if not get_backtestmanager().values.empty:

    st.write(get_backtestmanager().analytics.T)

    import plotly.graph_objects as go

    fig = go.Figure()
    for name, strategy in get_backtestmanager().strategies.items():
        # Add line chart for prices to the first subplot
        val = strategy.value.resample("M").last()
        price_trace = go.Scatter(
            x=val.index, y=val.values, name=name, hovertemplate="Date: %{x}<br>Price: %{y}"
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
        xaxis=dict(title="Date", tickformat="%Y-%m-%d"),  # Customize the date format
        yaxis=dict(
            title="Price", tickprefix="$"  # Add a currency symbol to the y-axis tick labels
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


for name, strategy in get_backtestmanager().strategies.items():
    with st.expander(label=name, expanded=False):
        st.button(
            label="Delete",
            key=name,
            on_click=get_backtestmanager().drop_strategy,
            kwargs={"name": name},
        )

        st.line_chart(
            pd.concat(
                [
                    strategy.value,
                    strategy.prices_bm,
                ],
                axis=1,
            )
        )
        st.line_chart(metrics.to_drawdown(strategy.value))
        st.bar_chart(strategy.allocations)
