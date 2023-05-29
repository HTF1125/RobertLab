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


def get_backtestmanager(start: str = "2006-1-1") -> BacktestManager:

    # Initialization
    if "backtestmanager" not in st.session_state:
        st.session_state["backtestmanager"] = BacktestManager.from_universe(
            start=start, commission=10, shares_frac=0
        )

    return st.session_state["backtestmanager"]


get_backtestmanager().Momentum(months=1)
get_backtestmanager().Momentum(months=3)
get_backtestmanager().Momentum(months=6)

st.line_chart(get_backtestmanager().values)
st.write("Complete.")
