from datetime import datetime, timedelta
import streamlit as st
from core.strategies import BacktestManager
from core import data


st.set_page_config(
    page_title="ROBERT'S WEBSITE",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)


def get_backtestmanager(start: str = "2007-1-1") -> BacktestManager:
    # Initialization
    if "backtestmanager" not in st.session_state:
        st.session_state["backtestmanager"] = BacktestManager(start=start)
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

    cols = st.columns([1, 1])
    return (
        cols[0].date_input(label="Start Date", value=start),
        cols[1].date_input(label="End Date", value=end),
    )


momentum_tab, base_tab = st.tabs(["Momentum", "Base"])

with momentum_tab:
    with st.form(key="momentum_month"):
        months = st.number_input(label="Momentum Months", min_value=1, max_value=36)
        objective = st.selectbox(
            label="Allocation Objective",
            options=[
                "uniform_allocation",
                "risk_parity",
                "minimized_correlation",
                "minimized_volatility",
                "maximized_sharpe_ratio",
            ],
        )
        frequency = st.selectbox(
            label="Rebalancing Frequency", options=["D", "M", "Q", "Y"]
        )
        absolute = st.checkbox(label="Absolute Momentum", value=False)
        s, e = getDateRange()
        submitted = st.form_submit_button("Submit")
        if submitted:
            get_backtestmanager().start = str(s)
            get_backtestmanager().end = str(e)
            get_backtestmanager().frequency = frequency
            get_backtestmanager().Momentum(
                months=months, objective=objective, absolute=absolute
            )


st.line_chart(get_backtestmanager().values)


st.table(get_backtestmanager().analytics.T)
