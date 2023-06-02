import streamlit as st
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


    (
        universe,
        objective,
        start,
        end,
        frequency,
        commission,
    ) = get_strategy_general_params()
    with st.form(key="momentum"):

        (
            months,
            skip_months,
            target_percentile,
            absolute,
        ) = get_startegy_momentum_params()


        submitted = st.form_submit_button("Submit")


        if submitted:
            get_backtestmanager().set_universe(name=universe)
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
                    target_percentile=target_percentile / 100.,
                )