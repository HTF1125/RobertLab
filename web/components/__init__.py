from typing import List
from datetime import datetime, timedelta
import streamlit as st


def get_start(default: datetime = datetime.today() - timedelta(days=10 * 365)):

    return st.date_input(label="Start Date", value=default)


def get_end(default: datetime = datetime.today()):

    return st.date_input(label="End Date", value=default)


def get_frequency(default: str = "M") -> str:
    options = ["D", "M", "Q", "Y"]
    return str(
        st.selectbox(
            label="Rebalancing Frequency",
            options=options,
            index=options.index(default),
            help="Select strategy's rebalancing frequency.",
        )
    )


def get_commission(default: int = 10, min_value=0, max_value=50, step=10) -> int:

    return int(
        st.number_input(
            label="Commission (bps)",
            min_value=min_value,
            max_value=max_value,
            step=step,
            value=default,
            help="Select strategy's trading commission in basis points.",
        )
    )


def get_objective(default: str = "uniform_allocation") -> str:
    options = [
        "uniform_allocation",
        "risk_parity",
        "minimized_correlation",
        "minimized_volatility",
        "maximized_sharpe_ratio",
    ]
    return str(
        st.selectbox(
            label="Rebalancing Frequency",
            options=options,
            index=options.index(default),
            help="Select strategy's allocation objective.",
        )
    )


def get_strategy_general_params():

    cols = st.columns([1] * 5)

    with cols[0]:

        s = get_start()

    with cols[1]:

        e = get_end()

    with cols[2]:

        o = get_objective()

    with cols[3]:

        f = get_frequency()

    with cols[4]:

        c = get_commission()

    st.markdown("---")
    return o, s, e, f, c
