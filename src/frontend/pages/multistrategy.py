"""ROBERT"""
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st
from src.backend.core import portfolios, strategies, benchmarks, universes
from src.backend.core.factors import single, MultiFactors
from .base import BasePage


class MultiStrategy(BasePage):
    def load_states(self) -> None:
        if "strategy" not in st.session_state:
            st.session_state["strategy"] = strategies.MultiStrategy()
        if "min_window" not in st.session_state:
            st.session_state["min_window"] = 252
        if "commission" not in st.session_state:
            st.session_state["commission"] = 10
        if "frequency" not in st.session_state:
            st.session_state["frequency"] = "M"
        if "optimizer" not in st.session_state:
            st.session_state["optimizer"] = "EqualWeight"
        if "inception" not in st.session_state:
            st.session_state["inception"] = "2003-01-01"
        if "allow_fractional_shares" not in st.session_state:
            st.session_state["allow_fractional_shares"] = False
        if "factors" not in st.session_state:
            st.session_state["factors"] = None

    @staticmethod
    def get_strategy() -> strategies.MultiStrategy:
        return st.session_state["strategy"]

    @staticmethod
    def get_inception() -> None:
        inception = str(
            st.date_input(
                label="Incep",
                value=pd.Timestamp(st.session_state["inception"]),
            )
        )
        st.session_state["inception"] = inception

    @staticmethod
    def get_optimizer() -> None:
        optimizer = str(
            st.selectbox(
                label="Opt",
                options=portfolios.__all__,
                index=portfolios.__all__.index(
                    st.session_state.get("optimizer", portfolios.__all__[0])
                ),
                help="Select strategy's rebalancing frequency.",
            )
        )
        st.session_state["optimizer"] = optimizer

    @staticmethod
    def get_benchmark() -> None:
        benchmark = str(
            st.selectbox(
                label="BM",
                options=benchmarks.__all__,
                index=benchmarks.__all__.index(
                    st.session_state.get("benchmark", benchmarks.__all__[0])
                ),
                help="Select strategy's rebalancing frequency.",
            )
        )
        st.session_state["benchmark"] = benchmark

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
    def get_min_window() -> None:
        min_window = int(
            st.number_input(
                label="Win",
                min_value=2,
                max_value=1500,
                step=100,
                value=st.session_state["min_window"],
                help="Minimum window of price data required.",
            )
        )
        st.session_state["min_window"] = min_window

    def get_strategy_parameters(self) -> None:
        set_parameter_funcs = [
            {
                "universe": self.get_universe,
                "optimizer": self.get_optimizer,
                "benchmark": self.get_benchmark,
                "frequency": self.get_frequency,
            },
            {
                "inception": self.get_inception,
                "commission": self.get_commission,
                "min_window": self.get_min_window,
            },
        ]
        for parameter_funcs in set_parameter_funcs:
            parameter_cols = st.columns([1] * len(parameter_funcs))

            for col, (name, func) in zip(parameter_cols, parameter_funcs.items()):
                with col:
                    func()

    @staticmethod
    def get_allow_fractional_shares() -> None:
        allow_fractional_shares = st.checkbox(
            label="Fractional Shares",
            value=False,
            help="Allow Fractional Shares Investing.",
        )
        st.session_state["allow_fractional_shares"] = allow_fractional_shares

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
                "max_value": 0.3,
                "step": 0.01,
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

    def get_factors(self) -> None:
        factors = tuple(
            st.multiselect(
                label="Factor List",
                options=single.__all__,
            )
        )


        st.session_state["factors"] = factors

    def load_page(self):
        with st.form("AssetAllocationForm"):
            # Backtest Parameters
            self.get_strategy_parameters()
            universe = getattr(universes, st.session_state["universe"])()
            # Factor Implementation
            self.get_factors()
            self.get_allow_fractional_shares()

            # Asset Allocation Constraints
            with st.expander(label="Custom Constraints:"):
                st.subheader("Optimizer Constraint")
                optimizer_constraints = self.get_optimizer_constraints()
                self.divider()
                st.subheader("Specific Constraint")
                specific_constraints = self.get_specific_constraints(
                    universe=universe.data
                )

            submitted = st.form_submit_button(label="Backtest", type="primary")

            if submitted:
                prices = self.get_universe_prices()

                in_signiture = {
                    "universe": st.session_state["universe"],
                    "strategy": {
                        "optimizer": st.session_state["optimizer"],
                        "benchmark": st.session_state["benchmark"],
                        "frequency": st.session_state["frequency"],
                        "inception": st.session_state["inception"],
                        "commission": st.session_state["commission"],
                        "min_window": st.session_state["min_window"],
                    },
                    "constraints": {
                        "factors": st.session_state["factors"],
                        "optimizer": optimizer_constraints,
                        "specific": specific_constraints,
                    },
                }

                with st.spinner(text="Backtesting in progress..."):
                    strategy = self.get_strategy().run(
                        **in_signiture["strategy"],
                        optimizer_constraints=optimizer_constraints,
                        specific_constraints=specific_constraints,
                        prices=prices,
                        factors=None
                        if not st.session_state["factors"]
                        else MultiFactors(
                            tickers=universe.tickers,
                            factors=st.session_state["factors"],
                        ).standard_percentile,
                    )
                    setattr(strategy, "signiture", in_signiture)

        multistrategy = self.get_strategy()

        if multistrategy:
            st.button(
                label="Clear All Strategies",
                on_click=self.get_strategy().clear,
            )

            analytics = multistrategy.analytics
            st.dataframe(analytics.T, use_container_width=True)

            st.plotly_chart(
                self.line(
                    data=self.get_strategy().performance.resample("M").last(),
                    yaxis_title="NAV",
                    yaxis_tickformat="$,.0f",
                    hovertemplate="Date: %{x} - Value: %{y:,.0f}",
                    title="Strategy Performance",
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            for name, strategy in multistrategy.items():
                with st.expander(label=name, expanded=False):
                    try:
                        st.json(getattr(strategy, "signiture"), expanded=False)
                    except KeyError:
                        st.warning("Signiture store not found.")

                    perf_tab, dd_tab, hw_tab, cw_tab = st.tabs(
                        ["Performance", "Drawdown", "Hist.Weights", "Curr.Weights"]
                    )

                    with perf_tab:
                        fig = self.line(
                            strategy.book.records.performance.to_frame(),
                            yaxis_title="NAV",
                            yaxis_tickformat="$,.0f",
                            hovertemplate="Date: %{x} - Value: %{y:,.0f}",
                            title="Strategy Performance",
                            legend_xanchor="left",
                            legend_y=1.1,
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
                    with dd_tab:
                        fig = self.line(
                            strategy.drawdown.to_frame(),
                            yaxis_title="Drawdwon",
                            yaxis_tickformat=".0%",
                            hovertemplate="Date: %{x} - Value: %{y:.2%}",
                            title="Strategy Drawdown",
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
                    with hw_tab:
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

                    with cw_tab:
                        fig = self.pie(
                            strategy.book.records.allocations.iloc[-1].dropna(),
                            title="Strategy Current Weights",
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
