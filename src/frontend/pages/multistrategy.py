"""ROBERT"""
from typing import Dict, List, Any, Callable
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.core import strategies
from .base import BasePage
from .. import components


class MultiStrategy(BasePage):
    def load_states(self) -> None:
        if "strategy" not in st.session_state:
            multistrategy = strategies.MultiStrategy().load_files()
            st.session_state["strategy"] = multistrategy

    @staticmethod
    def get_strategy() -> strategies.MultiStrategy:
        return st.session_state["strategy"]

    def clear_strategies(self):
        self.get_strategy().clear()

    def get_strategy_parameters(self) -> Dict[str, Any]:
        parameters = {}
        set_parameter_funcs = [
            {
                "inception": components.get_inception,
                "commission": components.get_commission,
                "min_window": components.get_min_window,
            },
        ]
        for parameter_funcs in set_parameter_funcs:
            parameter_cols = st.columns([1] * len(parameter_funcs))
            for col, (name, func) in zip(parameter_cols, parameter_funcs.items()):
                with col:
                    parameters[name] = func()

        return parameters

    def get_portfolio_constraints(self):
        constraints = {}
        kwargs = [
            {
                "name": "weight",
                "label": "Weight",
                "min_value": 0.0,
                "max_value": 2.0,
                "step": 0.02,
            },
            {
                "name": "return",
                "label": "Return",
                "min_value": 0.0,
                "max_value": 0.3,
                "step": 0.01,
            },
            {
                "name": "volatility",
                "label": "Volatility",
                "min_value": 0.0,
                "max_value": 0.3,
                "step": 0.01,
            },
            {
                "name": "active_weight",
                "label": "Act. Weight",
                "min_value": 0.0,
                "max_value": 1.0,
                "step": 0.05,
            },
            {
                "name": "expost_tracking_error",
                "label": "Ex-Post T.E.",
                "min_value": 0.0,
                "max_value": 0.1,
                "step": 0.01,
            },
            {
                "name": "exante_tracking_error",
                "label": "Ex-Ante T.E.",
                "min_value": 0.0,
                "max_value": 0.1,
                "step": 0.01,
            },
        ]

        cols = st.columns([1] * len(kwargs), gap="large")

        for idx, kwarg in enumerate(kwargs):
            assert isinstance(kwarg, dict)
            name = kwarg.pop("name")
            with cols[idx]:
                minimum, maximum = self.get_bounds(
                    **kwarg, format_func=lambda x: f"{x:.0%}"
                )
                if minimum is not None:
                    constraints[f"min_{name}"] = minimum
                if maximum is not None:
                    constraints[f"max_{name}"] = maximum

        return constraints

    def get_specific_constraints(
        self, universe: pd.DataFrame, num_columns: int = 5
    ) -> List[Dict]:
        constraints = []
        asset_classes = universe["assetclass"].unique()
        final_num_columns = min(num_columns, len(asset_classes))
        cols = st.columns([1] * final_num_columns, gap="large")
        for idx, asset_class in enumerate(asset_classes):
            with cols[idx % num_columns]:
                bounds = self.get_bounds(
                    label=asset_class,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    format_func=lambda x: f"{x:.0%}",
                )
                if bounds == (None, None):
                    continue
                constraint = {
                    "assets": universe[
                        universe["assetclass"] == asset_class
                    ].ticker.to_list(),
                    "bounds": bounds,
                }
                constraints.append(constraint)

        self.divider()
        final_num_columns = min(num_columns, len(universe))
        cols = st.columns([1] * final_num_columns, gap="large")
        for idx, asset in enumerate(universe.to_dict("records")):
            ticker = asset["ticker"]
            name = asset["name"]
            with cols[idx % num_columns]:
                bounds = self.get_bounds(
                    label=ticker,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    format_func=lambda x: f"{x:.0%}",
                    help=name,
                )
                if bounds == (None, None):
                    continue
                constraint = {
                    "assets": ticker,
                    "bounds": bounds,
                }
                constraints.append(constraint)
        return constraints

    @staticmethod
    def get_horizon_input(*funcs: Callable) -> None:
        cols = st.columns(len(funcs))

        for col, func in zip(cols, funcs):
            with col:
                func()

    def load_page(self):
        self.get_horizon_input(components.get_universe, components.get_states)

        with st.expander(label="State"):
            state = components.Session.get_state()
            selected_state = st.selectbox(label="Select state", options=state.states)

            self.get_horizon_input(
                components.get_portfolio,
                components.get_factor,
            )
            portfolio_constraints = self.get_portfolio_constraints()

        with st.form("AssetAllocationForm"):
            # portfolio_constraints = self.get_portfolio_constraints()

            self.get_horizon_input(
                components.get_frequency,
                components.get_inception,
                components.get_commission,
                components.get_min_window,
            )
            allow_fractional_shares = components.get_allow_fractional_shares()



            # with st.expander(label="Custom Constraints:", expanded=False):
            #     specific_constraints = self.get_specific_constraints(
            #         universe=pd.DataFrame(components.Session.get_universe().ASSETS)
            #     )

            submitted = st.form_submit_button(label="Backtest", type="primary")

            if submitted:
                # prices = get_prices(tickers=universe.get_tickers())

                with st.spinner(text="Backtesting in progress..."):
                    self.get_strategy().add_strategy(
                        universe=components.Session.get_universe(),
                        portfolio=components.Session.get_portfolio(),
                        frequency=components.Session.get_frequency(),
                        factor=components.Session.get_factor(),
                        inception=str(components.Session.get_inception()),
                        commission=components.Session.get_commission(),
                        min_window=components.Session.get_min_window(),
                        allow_fractional_shares=allow_fractional_shares,
                        portfolio_constraints=portfolio_constraints,
                        # specific_constraints=specific_constraints,
                    )

        multistrategy = self.get_strategy()

        if multistrategy:
            st.button(label="Clear Strategies", on_click=self.clear_strategies)

            # fig = go.Figure()

            # for name, strategy in multistrategy.items():
            #     performance = strategy.performance
            #     num_points = len(performance)

            #     indices = np.linspace(0, num_points - 1, 10, dtype=int)

            #     performance = performance.iloc[indices]

            #     fig.add_trace(
            #         go.Scatter(
            #             x=performance.index,
            #             y=performance.values,
            #             name=name,
            #         )
            #     )

            # fig.update_layout(
            #     title="Performance", legend_orientation="h", hovermode="x unified"
            # )
            # st.plotly_chart(
            #     fig,
            #     use_container_width=True,
            #     config={"displayModeBar": False},
            # )
            st.write(multistrategy.analytics.T)

        # components.plot_multistrategy(multistrategy, allow_save=True)
