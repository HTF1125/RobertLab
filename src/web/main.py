from datetime import datetime, timedelta
import streamlit as st
from core.strategies import BacktestManager
from core import data
from core import metrics
import pandas as pd


st.set_page_config(
    page_title="ROBERT'S WEBSITE",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)


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


def getDateRange(
    start: datetime = datetime.today() - timedelta(days=10 * 365),
    end: datetime = datetime.today(),
):
    cols = st.columns([1, 1, 1])
    return (
        cols[0].date_input(label="Start Date", value=start),
        cols[1].date_input(label="End Date", value=end),
        cols[2].select_slider(
            label="Commission (bps)", value=10, options=range(0, 100 + 10, 10)
        ),
    )


def clear_strategies():
    get_backtestmanager().reset_strategies()


momentum_tab, base_tab = st.tabs(["Momentum", "Base"])

with momentum_tab:
    st.button(label="Clear Strategies", on_click=clear_strategies)

    with st.form(key="momentum_month"):
        cols = st.columns([1, 1, 1, 1])

        months = cols[0].select_slider(
            label="Momentum Months", options=range(1, 36 + 1), value=1
        )
        skip_months = cols[1].select_slider(
            label="Momentum Skip Months", options=range(0, 6 + 1), value=0
        )
        frequency = cols[2].select_slider(
            label="Rebalancing Frequency", options=["D", "M", "Q", "Y"], value="D"
        )
        target_percentile = cols[3].select_slider(
            label="Target Percentile",
            options=range(0, 100 + 10, 10),
            value=70,
        )
        objective = st.radio(
            label="Allocation Objective",
            options=[
                "uniform_allocation",
                "risk_parity",
                "minimized_correlation",
                "minimized_volatility",
                "maximized_sharpe_ratio",
            ],
            horizontal=True,
        )
        absolute = st.checkbox(label="Absolute Momentum", value=False)
        c1, c2 = st.columns(
            [
                1,
                1,
            ]
        )
        s, e = c1.select_slider(
            label="Select backtest date range",
            options=get_backtestmanager().prices.index,
            format_func=lambda x: format(x, "%Y-%m-%d"),
            value=(
                get_backtestmanager().prices.index[0],
                get_backtestmanager().prices.index[-1],
            ),
        )
        c = c2.select_slider(
            label="Commission (bps)", value=10, options=range(0, 100 + 10, 10)
        )

        submitted = st.form_submit_button("Submit")
        if submitted:
            get_backtestmanager().commission = int(c)
            get_backtestmanager().start = str(s)
            get_backtestmanager().end = str(e)
            get_backtestmanager().frequency = frequency
            get_backtestmanager().Momentum(
                months=months,
                skip_months=skip_months,
                objective=objective,
                absolute=absolute,
                target_percentile=target_percentile / 100,
            )


if not get_backtestmanager().values.empty:
    st.line_chart(get_backtestmanager().values.resample("M").last())
    st.write(get_backtestmanager().analytics.T)


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
