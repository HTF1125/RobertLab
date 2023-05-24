import streamlit as st
import plotly.graph_objects as go
import database as db


st.set_page_config(
    page_title="ROBERT'S WEBSITE",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)


st.write("Robert's Website")

data = db.data.get_us_pce().pct_change(12)
fig = go.Figure(data=go.Scatter(x=data.index, y=data.iloc[:, 0], mode='lines'))
st.plotly_chart(fig)


data = db.data.get_us_cpi_all_urban().pct_change(12)
fig = go.Figure(data=go.Scatter(x=data.index, y=data.iloc[:, 0], mode='lines'))
st.plotly_chart(fig)

data = db.data.get_oecd_us_leading_indicator() - 100
fig = go.Figure(data=go.Scatter(x=data.index, y=data.iloc[:, 0], mode='lines'))
st.plotly_chart(fig)
