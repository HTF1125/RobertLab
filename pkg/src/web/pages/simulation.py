import streamlit as st


def main():
    investment_horizon = st.select_slider(
        label="Investment Horizon",
        options=range(10, 51, 1),
        value=20,
    )

    initial_investment = st.select_slider(
        label="Initial Investment",
        options=range(10_000, 1_001_000, 1_000),
        value=10_000,
    )

    monthly_investment = st.select_slider(
        label="Monthly Investment",
        options=range(1_000, 1_001_000, 1_000),
        value=1_000,
    )
