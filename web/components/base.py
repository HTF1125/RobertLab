"""ROBERT"""
from datetime import datetime, timedelta
import streamlit as st
from .. import state


def get_date_range():
    left, right = st.columns([1, 1])
    with left:
        s = get_start()
    with right:
        e = get_end()
    return s, e


def get_start() -> str:

    return str(
        st.date_input(
            label="Start Date",
            value=datetime.today() - timedelta(days=3650),
            help="Start date of the strategy backtest.",
        )
    )


def get_end() -> str:

    return str(
        st.date_input(
            label="End Date",
            value=datetime.today(),
            help="End date of the strategy backtest.",
        )
    )


def get_frequency() -> str:
    options = ["D", "M", "Q", "Y"]
    return str(
        st.selectbox(
            label="Rebalancing Frequency",
            options=options,
            index=options.index("M"),
            help="Select strategy's rebalancing frequency.",
        )
    )


def get_commission() -> int:

    return int(
        st.number_input(
            label="Commission (bps)",
            min_value=0,
            max_value=100,
            step=10,
            value=10,
            help="Select strategy's trading commission in basis points.",
        )
    )


def get_objective() -> str:

    options = [
        "uniform_allocation",
        "risk_parity",
        "minimized_correlation",
        "minimized_volatility",
        "inverse_variance",
        "maximized_sharpe_ratio",
        "hierarchical_risk_parity",
        "hierarchical_equal_risk_contribution",
    ]
    return str(
        st.selectbox(
            label="Allocation Objective",
            options=options,
            index=options.index("uniform_allocation"),
            format_func=lambda x: x.replace("_", " ").title().replace(" ", ""),
            help="Select strategy's rebalancing frequency.",
        )
    )


def get_universe():

    options = ["USSECTORETF", "GENERAL"]

    return st.selectbox(
        label="Investment Universe",
        options=options,
        index=options.index("USSECTORETF"),
        help="Select strategy's investment universe.",
    )


def get_strategy_general_params():

    task = [
        get_universe,
        get_start,
        get_end,
        get_objective,
        get_frequency,
        get_commission,
    ]

    cache = {}
    for idx, col in enumerate(st.columns([1] * len(task))):
        with col:
            cache[str(task[idx].__name__).replace("get_", "")] = task[idx]()
    return cache
