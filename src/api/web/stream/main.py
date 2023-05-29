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

# bt = BacktestManager.from_universe(start="2010-1-1", commission=10, shares_frac=0)
# st.write(bt.prices)
# bt.MinCorr()
# bt.EqualWeight()
# st.line_chart(bt.values)

# st.write(bt.analytics)


st.line_chart(data.get_macro(tickers="USALOLITONOSTSAM"))



st.line_chart(data.get_macro(tickers="BAMLC0A1CAAAEY"))


__cpi__ ={
        "CPIAUCSL": "CPI-All",
        "CPILFESL": "CPI-Core",
        "PPIACO" : "PPI-All",
        "WPU10": "PPI-Metal",
        "USALOLITONOSTSAM": "OECEUSLEI",
        "BAMLC0A1CAAAEY": "CreditYieldAAA",
        "T10Y2Y": "USL10YS2Y",
        "MORTGAGE30US": "US30YMortgage",
        "FEDFUNDS": "Fed.Effective.Rate",
        "WALCL": "TotalAsset",
        "T10Y3M" : "USL10YS3M",
        "BAMLH0A0HYM2": "US.HY.OAS",
        "UNRATE": "US.Unemployment.Rate",

    }

cpi = data.get_macro(tickers = __cpi__)

st.line_chart(cpi.pct_change(12).loc["2010":])



