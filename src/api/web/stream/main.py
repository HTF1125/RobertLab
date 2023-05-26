import streamlit as st
from core.strategies import BacktestManager


st.set_page_config(
    page_title="ROBERT'S WEBSITE",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)


st.write("test")

bt = BacktestManager.from_universe(start="2005-1-1", commission=10, shares_frac=0)
# bt.EqualWeight()
# bt.MinCorr()
# bt.RegimeRotation(signal=signal)
# bt.EqualWeight(prices=yf.download("SPY")["Adj Close"].to_frame())
# bt.RegimeRotationMinCorr(signal=signal)
# bt.RegimeRotationMinCorr(signal=signal, span=21*3)
# bt.MinVol(span=21*12)
# bt.MMM()
bt.MeanReversion()
