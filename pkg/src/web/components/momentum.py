"""ROBERT"""

import streamlit as st
from pkg.src.core import data
from .base import get_strategy_general_params
from .. import state


def get_months() -> int:

    return int(
        st.select_slider(
            label="Momentum Months",
            options=range(1, 37, 1),
            value=1,
            help="Number of months to calculate momentum",
        )
    )


def get_skip_months() -> int:

    return int(
        st.select_slider(
            label="Momentum Skip Months",
            options=range(0, 7, 1),
            value=0,
            help="Number of months to skip for momentum",
        )
    )


def get_target_percentile() -> float:

    return (
        int(
            st.select_slider(
                label="Target Momentum Percentile",
                options=range(0, 110, 10),
                value=70,
                help="Target portfolio momentum score percentile.",
            )
        )
        / 100.0
    )


def get_absolute() -> bool:

    return st.checkbox(
        label="Aboslute", value=False, help="Use absolute momentum."
    )


def get_startegy_momentum_params():

    task = [
        get_months,
        get_skip_months,
        get_target_percentile,
    ]
    cache = {}
    for idx, col in enumerate(st.columns([1] * len(task))):
        with col:
            cache[str(task[idx].__name__)[4:]] = task[idx]()

    cache["absolute"] = get_absolute()
    return cache


def main():

    with st.form(key="momentum"):
        strategy_general_params = get_strategy_general_params()
        startegy_momentum_params = get_startegy_momentum_params()

        submitted = st.form_submit_button("Submit")

        if submitted:
            with st.spinner("loading prices data ..."):
                universe = strategy_general_params.pop("universe")
                prices = data.get_prices(
                    "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU"
                )
            state.strategy.get_backtestmanager().prices = prices
            with st.spinner(text="Backtesting in progress..."):
                state.strategy.get_backtestmanager().Momentum(
                    **strategy_general_params, **startegy_momentum_params
                )
    st.button(
        label="Clear All Strategies",
        on_click=state.strategy.get_backtestmanager().reset_strategies,
    )
