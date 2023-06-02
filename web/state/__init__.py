import streamlit as st
from pkg.src.core.strategies import BacktestManager


def get_backtestmanager() -> BacktestManager:
    # Initialization
    if "backtestmanager" not in st.session_state:
        st.session_state["backtestmanager"] = BacktestManager()
    return st.session_state["backtestmanager"]
