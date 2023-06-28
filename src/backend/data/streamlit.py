"""ROBERT"""

try:
    import streamlit as st
except ImportError as exc:
    raise ImportError("Unable to import streamlit.") from exc
import pandas as pd
from . import base


@st.cache_data(ttl="1d")
def get_price(ticker: str) -> pd.DataFrame:
    return base.get_price(ticker)
