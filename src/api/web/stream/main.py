from datetime import datetime
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


def get_backtestmanager(start: str = "2023-1-1") -> BacktestManager:
    # Initialization
    if "backtestmanager" not in st.session_state:
        st.session_state["backtestmanager"] = BacktestManager(start=start)
    return st.session_state["backtestmanager"]


st.session_state["universe"] = st.selectbox(
    label="Select Investment Universe",
    options=["USSECTORETF", "General"],
)
get_backtestmanager().set_universe(name=st.session_state["universe"])


with st.form(key="strategy", clear_on_submit=False):
    c1, c2 = st.columns([1, 1])
    st.session_state["start"] = c1.date_input(label="start", value=datetime(2006, 1, 1))
    st.session_state["end"] = c2.date_input(label="end", value=datetime.today())

    submitted = st.form_submit_button("Submit")

    if submitted:
        get_backtestmanager().Momentum(months=3)
        # get_backtestmanager().Momentum(months=3)
        # get_backtestmanager().Momentum(months=6)


st.write(get_backtestmanager().prices)
st.write(get_backtestmanager().analytics)
st.line_chart(get_backtestmanager().values)

# st.bar_chart(get_backtestmanager().strategies["Momentum(months=1)"].allocations)
