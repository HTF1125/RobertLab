import streamlit as st
from pkg.src.core import data
from .base import get_strategy_general_params
from ..state import get_backtestmanager



def get_startegy_momentum_params():

    cols = st.columns([1] * 3)

    months = cols[0].select_slider(
        label="Momentum Months", options=range(1, 36 + 1), value=1
    )
    skip_months = cols[1].select_slider(
        label="Momentum Skip Months", options=range(0, 6 + 1), value=0
    )

    target_percentile = cols[2].select_slider(
        label="Target Percentile",
        options=range(0, 100 + 10, 10),
        value=70,
    )

    absolute = st.checkbox(label="Absolute Momentum", value=False)

    return months, skip_months, target_percentile, absolute


def main():
    
    with st.form(key="momentum"):
        (
            universe,
            objective,
            start,
            end,
            frequency,
            commission,
        ) = get_strategy_general_params()
        (
            months,
            skip_months,
            target_percentile,
            absolute,
        ) = get_startegy_momentum_params()


        submitted = st.form_submit_button("Submit")


        if submitted:
            prices = data.get_prices("XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU")
            get_backtestmanager().prices = prices
            get_backtestmanager().commission = int(commission)
            get_backtestmanager().start = str(start)
            get_backtestmanager().end = str(end)
            get_backtestmanager().frequency = frequency
            with st.spinner(text="Backtesting in progress..."):

                get_backtestmanager().Momentum(
                    months=months,
                    skip_months=skip_months,
                    objective=objective,
                    absolute=absolute,
                    target_percentile=int(target_percentile) / 100.,
                )