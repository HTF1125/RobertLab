"""ROBERT"""

import numpy as np
import pandas as pd
import streamlit as st
from src.backend.core import metrics
from src.backend.misc import charts
from src.backend.web import data
from ..components import get_unive


def get_investment_horizon():
    return st.number_input(
        label="Investment Horizon",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
    )


def get_initial_investment():
    return st.number_input(
        label="Initial Investment",
        min_value=0,
        max_value=1_000_000,
        step=10_000,
        value=100_000,
        # format = ","
    )


def get_monthly_investment():
    return st.number_input(
        label="Monthly Investment",
        min_value=0,
        max_value=1_000_000,
        step=10_000,
        value=100_000,
    )


from typing import Dict, Callable


def get_multi_parameters(parameter_funcs: Dict[str, Callable]):
    parameters = {}
    parameter_cols = st.columns([1] * len(parameter_funcs))

    for col, (name, func) in zip(parameter_cols, parameter_funcs.items()):
        with col:
            parameters[name] = func()
    return parameters





def main():
    universe = get_unive()

    st.write(universe)



    with st.form(key="simulatioin"):
        parameters = get_multi_parameters(
            {
                "initial_investment": get_initial_investment,
                "monthly_investment": get_monthly_investment,
                "investment_horizon": get_investment_horizon,
            },
        )

        num_simulations = st.number_input(
            label="Number of Simulations",
            min_value=10_000,
            max_value=100_000,
            step=10_000,
            value=10_000,
        )

        total_investment = (
            parameters["initial_investment"]
            + parameters["monthly_investment"] * 12 * parameters["investment_horizon"]
        )
        investment_goal = st.number_input(
            label="Investment Goal",
            min_value=total_investment,
            step=100_000,
            max_value=total_investment * 2,
            value=total_investment,
        )

        submitted = st.form_submit_button("Simulate")

        if submitted:
            prices = (
                data.get_prices(tickers=universe.ticker.tolist()).resample("M").last()
            )
            n_points = 12 * parameters["investment_horizon"]

            log_return = metrics.to_log_return(prices=prices)
            expected_return = log_return.mean()
            cov = log_return.cov()
            corr_cov = np.linalg.cholesky(cov)
            z = np.random.normal(
                0, 1, size=(len(prices.columns), n_points * num_simulations)
            )

            drift = np.full(
                (n_points * num_simulations, len(prices.columns)), expected_return
            ).T
            shock = np.dot(corr_cov, z)

            monthly_returns = drift + shock

            f_monthly_returns = np.dot(
                monthly_returns.T, pd.Series(dict(SPY=0.6, AGG=0.2, TLT=0.2))
            )
            f_monthly_returns = f_monthly_returns.reshape(n_points, num_simulations)

            result = pd.DataFrame(
                np.exp(f_monthly_returns),
                # columns=prices.columns,
                index=pd.date_range(start="2023-6-8", periods=int(n_points), freq="M"),
            ).cumprod()

            st.plotly_chart(
                charts.create_lineplot(result.iloc[:, :100]), use_container_width=True
            )

            # charts.create_lineplot(result.median(axis=1).to_frame())
