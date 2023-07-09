"""ROBERT"""
from typing import Dict, List, Any, Callable
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.core import strategies, portfolios, factors
from .base import BasePage
from .. import components


class MultiStrategy(BasePage):
    def load_states(self) -> None:
        if not "local_multistrategy" in st.session_state:
            st.session_state["local_multistrategy"] = strategies.MultiStrategy()

    @staticmethod
    def get_strategy() -> strategies.MultiStrategy:
        return st.session_state["local_multistrategy"]

    def clear_strategies(self):
        self.get_strategy().clear()

    def load_page(self):
        multistrategy = self.get_strategy()
        universe = components.get_universe()
        regime = components.get_regime()

        # with st.expander(label="Custom Constraints:", expanded=False):
        constraint = {}
        if regime.__states__:
            for col, state in zip(
                st.columns(len(regime.__states__)), regime.__states__
            ):
                with col:
                    with st.expander(state, expanded=True):
                        state_constraint = {}
                        constraintSet = components.StrategyConstraint(prefix=state)
                        state_constraint["portfolio_constraint"] = constraintSet.dict()

                        state_constraint[
                            "asset_constraint"
                        ] = constraintSet.get_asset_constraints(
                            universe=pd.DataFrame(universe.ASSETS)
                        )
                        constraint[state] = state_constraint

        with st.form("AssetAllocationForm"):
            params = components.StrategyParameters().dict()

            submitted = st.form_submit_button(label="Backtest", type="primary")

            if submitted:
                with st.spinner(text="Backtesting in progress..."):
                    multistrategy.add_strategy(
                        universe=universe,
                        regime=regime,
                        constraint=constraint,
                        **params,
                    )

        if multistrategy:
            st.button(label="Clear Strategies", on_click=multistrategy.clear)

            fig = go.Figure()

            for name, strategy in multistrategy.items():
                performance = strategy.performance / strategy.initial_investment - 1
                num_points = len(performance)
                indices = np.linspace(0, num_points - 1, 30, dtype=int)
                performance = performance.iloc[indices]
                fig.add_trace(
                    go.Scatter(
                        x=performance.index,
                        y=performance.values,
                        name=name,
                    )
                )

            fig.update_layout(
                title="Performance",
                legend_orientation="h",
                hovermode="x unified",
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.write(multistrategy.analytics.T)

        components.plot_multistrategy(multistrategy, allow_save=True)
