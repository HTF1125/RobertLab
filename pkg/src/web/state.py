"""ROBERT"""
import streamlit as st
from pkg.src.core.strategies import MultiStrategy


def get_multistrategy() -> MultiStrategy:
    # Initialization
    if "multi-strategy" not in st.session_state:
        st.session_state["multi-strategy"] = MultiStrategy()
    return st.session_state["multi-strategy"]

