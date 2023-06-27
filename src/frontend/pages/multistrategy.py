"""ROBERT"""
from typing import Dict, List, Tuple, Any
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.core import portfolios, strategies, benchmarks, universes, factors
from src.backend.data import get_prices
from src.core.strategies.parser import Parser
from .base import BasePage


def save_strategy(name: str, strategy: strategies.Strategy):
    try:
        strategy.save(name)
        st.info(f"save strategy {name} complete.")
    except:
        st.warning(f"save file failed.")


class MultiStrategy(BasePage):
    def load_states(self) -> None:
        if "strategy" not in st.session_state:
            st.session_state["strategy"] = strategies.MultiStrategy()

    @staticmethod
    def get_strategy() -> strategies.MultiStrategy:
        return st.session_state["strategy"]

    @staticmethod
    def get_inception() -> str:
        return str(
            st.date_input(
                label="Incep",
                value=pd.Timestamp("2003-01-01"),
            )
        )

    @staticmethod
    def get_optimizer() -> str:
        return str(
            st.selectbox(
                label="Opt",
                options=portfolios.__all__,
                help="Select strategy's rebalancing frequency.",
            )
        )

    @staticmethod
    def get_benchmark() -> str:
        return str(
            st.selectbox(
                label="BM",
                options=benchmarks.__all__,
                help="Select strategy's rebalancing frequency.",
            )
        )

    @staticmethod
    def get_frequency() -> str:
        options = ["D", "M", "Q", "Y"]
        return str(
            st.selectbox(
                label="Freq",
                options=options,
                index=options.index("M"),
                help="Select strategy's rebalancing frequency.",
            )
        )

    @staticmethod
    def get_commission() -> int:
        return int(
            st.number_input(
                label="Comm",
                min_value=0,
                max_value=100,
                step=10,
                value=10,
                help="Select strategy's trading commission in basis points.",
            )
        )

    @staticmethod
    def get_min_window() -> int:
        return int(
            st.number_input(
                label="Win",
                min_value=2,
                max_value=1500,
                step=100,
                value=252,
                help="Minimum window of price data required.",
            )
        )

    def get_strategy_parameters(self) -> Dict[str, Any]:
        parameters = {}
        set_parameter_funcs = [
            {
                "universe": self.get_universe,
                "benchmark": self.get_benchmark,
                "optimizer": self.get_optimizer,
                "frequency": self.get_frequency,
            },
            {
                "inception": self.get_inception,
                "commission": self.get_commission,
                "min_window": self.get_min_window,
            },
            {
                "factors": self.get_factors,
            },
            {"allow_fractional_shares": self.get_allow_fractional_shares},
        ]
        for parameter_funcs in set_parameter_funcs:
            parameter_cols = st.columns([1] * len(parameter_funcs))

            for col, (name, func) in zip(parameter_cols, parameter_funcs.items()):
                with col:
                    parameters[name] = func()

        return parameters

    @staticmethod
    def get_allow_fractional_shares() -> bool:
        return st.checkbox(
            label="Fractional Shares",
            value=False,
            help="Allow Fractional Shares Investing.",
        )

    def get_optimizer_constraints(self):
        constraints = {}
        kwargs = [
            {
                "name": "weight",
                "label": "Weight",
                "min_value": 0.0,
                "max_value": 1.0,
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

    def get_factors(self) -> Tuple[str]:
        return tuple(
            st.multiselect(
                label="Factor List",
                options=factors.__all__,
            )
        )

    def load_page(self):
        parameters = self.get_strategy_parameters()
        universe = Parser.get_universe(parameters["universe"])
        with st.form("AssetAllocationForm"):
            # Backtest Parameters
            # Asset Allocation Constraints
            optimizer_constraints = self.get_optimizer_constraints()
            with st.expander(label="Custom Constraints:"):
                st.subheader("Specific Constraint")
                specific_constraints = self.get_specific_constraints(
                    universe=pd.DataFrame(universe.ASSETS)
                )

            submitted = st.form_submit_button(label="Backtest", type="primary")

            if submitted:
                prices = get_prices(tickers=universe.get_tickers())

                with st.spinner(text="Backtesting in progress..."):
                    strategy = self.get_strategy().run(
                        prices=prices,
                        **parameters,
                        optimizer_constraints=optimizer_constraints,
                        specific_constraints=specific_constraints,
                    )

        multistrategy = self.get_strategy()

        if multistrategy:
            st.button(
                label="Clear All Strategies",
                on_click=self.get_strategy().clear,
            )

            analytics = multistrategy.analytics
            st.dataframe(analytics.T, use_container_width=True)

            fig = go.Figure()
            for name, strategy in multistrategy.items():
                performance = strategy.book.records.performance.resample("M").last()
                fig.add_trace(
                    go.Scatter(
                        x=performance.index,
                        y=performance.values,
                        hovertemplate="Date: %{x}: %{y}}",
                        name=name,
                    )
                )

            fig.update_layout(
                title="Strategy Performance",
                yaxis_tickformat=",.0f",
                hovermode="x unified",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

            for name, strategy in multistrategy.items():
                with st.expander(label=name, expanded=False):
                    strategy_name = st.text_input(
                        label="Name your strategy", placeholder="...", key=name
                    )

                    st.button(
                        label="Save",
                        on_click=save_strategy,
                        kwargs={"name": strategy_name, "strategy": strategy},
                        key=f"{name} save button",
                    )

                    try:
                        st.json(multistrategy.get_signature(name), expanded=False)
                    except KeyError:
                        st.warning("Signiture store not found.")

                    performance_tab, allocations_tab = st.tabs(
                        ["Performance", "Allocations"]
                    )

                    with performance_tab:
                        performance = strategy.book.records.performance

                        fig = go.Figure().add_trace(
                            go.Scatter(
                                x=performance.index,
                                y=performance.values,
                                name="Performance",
                                hovertemplate="Date: %{x} - Value: %{y}",
                            )
                        )

                        if not strategy.benchmark is None:
                            fig.add_trace(
                                go.Scatter(
                                    x=strategy.benchmark.performance.index,
                                    y=strategy.benchmark.performance.values,
                                    hovertemplate="Date: %{x}: %{y}}",
                                    name="benchmark",
                                )
                            )

                        fig.update_layout(title="Performance")
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )

                        drawdown = strategy.drawdown
                        fig = (
                            go.Figure()
                            .add_trace(
                                go.Scatter(
                                    x=drawdown.index,
                                    y=drawdown.values,
                                    name="Drawdown",
                                    hovertemplate="Date: %{x} - Value: %{y}",
                                )
                            )
                            .update_layout(
                                title="Drawdown",
                                hovermode="x unified",
                                xaxis_tickformat="%Y-%m-%d",
                                yaxis_tickformat=".2%",
                            )
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )

                    with allocations_tab:
                        fig = self.line(
                            strategy.book.records.allocations,
                            xaxis_tickformat="%Y-%m-%d",
                            xaxis_title="Date",
                            yaxis_title="Weights",
                            yaxis_tickformat=".0%",
                            hovertemplate="Date: %{x} - Value: %{y:.2%}",
                            title="Strategy Historical Weights",
                            stackgroup="stack",
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )

                        fig = self.pie(
                            strategy.book.records.allocations.iloc[-1].dropna(),
                            title="Strategy Current Weights",
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
