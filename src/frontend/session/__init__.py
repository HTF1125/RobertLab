import streamlit as st
from src.core import strategies


def getPage() -> str:
    if "page" not in st.session_state:
        st.session_state["page"] = "Dashboard"
    return st.session_state["page"]


def setPage(page: str = "Dashboard", rerun: bool = True):
    if st.session_state["page"] != page:
        st.session_state["page"] = page
        if rerun:
            st.experimental_rerun()


def getPreStrategy(
    load_files: bool = True, key: str = "pre-strategy"
) -> strategies.MultiStrategy:
    if key not in st.session_state:
        multistrategy = strategies.MultiStrategy()
        if load_files:
            multistrategy.load_files()
        st.session_state[key] = multistrategy
        return multistrategy
    return st.session_state[key]


def getNewStrategy(
    load_files: bool = True, key: str = "new-strategy"
) -> strategies.MultiStrategy:
    if key not in st.session_state:
        multistrategy = strategies.MultiStrategy()
        if load_files:
            multistrategy.load_files()
        st.session_state[key] = multistrategy
        return multistrategy
    return st.session_state[key]
