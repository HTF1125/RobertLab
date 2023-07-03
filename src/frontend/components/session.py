
import pandas as pd
import streamlit as st
from src.core import universes, portfolios, factors, states


class Session:
    @staticmethod
    def get_universe() -> universes.Universe:
        return universes.get(st.session_state["universe"])


    @staticmethod
    def get_portfolio() -> portfolios.Portfolio:
        return portfolios.get(st.session_state["portfolio"])


    @staticmethod
    def get_frequency() -> str:
        return st.session_state.get("frequency", "M")

    @staticmethod
    def get_factor() -> factors.MultiFactor:
        return factors.MultiFactor(*st.session_state.get("factors", ()))


    @staticmethod
    def get_commission() -> int:
        return st.session_state["commission"]

    @staticmethod
    def get_inception() -> pd.Timestamp:
        return st.session_state["inception"]

    @staticmethod
    def get_min_window() -> int:
        return st.session_state["min_window"]


    @staticmethod
    def get_state() -> states.State:
        return states.FixedTwoState()