"""ROBERT"""
from typing import Dict, List
import pandas as pd
import streamlit as st
from pkg.src.core import factors
from pkg.src.web import components, data
from pkg.src.core import factors, portfolios, strategies
from .base import BasePage


class InvestmentStrategy(BasePage):
    signiture = {}

    def __init__(self) -> None:
        if "multistrategy" not in st.session_state:
            st.session_state["multistrategy"] = strategies.MultiStrategy()
        super().__init__()

    def get_multistrategy(self) -> strategies.MultiStrategy:
        return st.session_state["multistrategy"]

    def get_name(self) -> str:
        return str(
            st.text_input(
                label="Name",
                value=f"Strategy-{self.get_multistrategy().num_strategies + 1}",
            )
        )

    def clear_strategy(self) -> None:
        self.signiture = {}
        multistrategy = self.get_multistrategy()
        multistrategy.strategies = {}
        multistrategy.num_strategies = 0

    def del_strategy(self, name: str) -> None:
        strategies = self.get_multistrategy().strategies
        del strategies[name]
        del self.signiture[name]

    @staticmethod
    def get_optimizer() -> str:
        optimizer = st.selectbox(
            label="Opt",
            options=[
                portfolios.EqualWeight.__name__,
                portfolios.MaxReturn.__name__,
                portfolios.MaxSharpe.__name__,
                portfolios.MinVolatility.__name__,
                portfolios.MinCorrelation.__name__,
                portfolios.InverseVariance.__name__,
                portfolios.RiskParity.__name__,
                portfolios.HRP.__name__,
                portfolios.HERC.__name__,
            ],
            help="Select strategy's rebalancing frequency.",
        )

        if optimizer is None:
            raise ValueError()
        return optimizer

    @staticmethod
    def get_benchmark() -> str:
        benchmark = st.selectbox(
            label="BM",
            options=[
                strategies.benchmarks.Global64.__name__,
                strategies.benchmarks.UnitedStates64.__name__,
            ],
            help="Select strategy's benchmark.",
        )

        if benchmark is None:
            raise ValueError()
        return benchmark

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

    def get_strategy_parameters(self):
        parameter_funcs = {
            "name" : self.get_name,
            "optimizer": self.get_optimizer,
            "benchmark": self.get_benchmark,
            "frequency": self.get_frequency,
            "commission": self.get_commission,
        }

        parameter_cols = st.columns([1] * len(parameter_funcs))

        parameters = {}

        for col, (name, func) in zip(parameter_cols, parameter_funcs.items()):
            with col:
                parameters[name] = func()

        return parameters

    @staticmethod
    def get_allow_fractional_shares():
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
                "step": 0.05,
                "format_func": lambda x: f"{x:.0%}",
            },
            {
                "name": "return",
                "label": "Return",
                "min_value": 0.0,
                "max_value": 0.3,
                "step": 0.01,
                "format_func": lambda x: f"{x:.0%}",
            },
            {
                "name": "volatility",
                "label": "Volatility",
                "min_value": 0.0,
                "max_value": 0.3,
                "step": 0.01,
                "format_func": lambda x: f"{x:.0%}",
            },
            {
                "name": "active_weight",
                "label": "Active Weight",
                "min_value": 0.0,
                "max_value": 0.3,
                "step": 0.01,
                "format_func": lambda x: f"{x:.0%}",
            },
            {
                "name": "expost_tracking_error",
                "label": "Ex-Post T.E.",
                "min_value": 0.0,
                "max_value": 0.1,
                "step": 0.01,
                "format_func": lambda x: f"{x:.0%}",
            },
            {
                "name": "exante_tracking_error",
                "label": "Ex-Ante T.E.",
                "min_value": 0.0,
                "max_value": 0.1,
                "step": 0.01,
                "format_func": lambda x: f"{x:.0%}",
            },
        ]

        cols = st.columns([1] * len(kwargs), gap="large")

        for idx, kwarg in enumerate(kwargs):
            assert isinstance(kwarg, dict)
            name = kwarg.pop("name")
            with cols[idx]:
                minimum, maximum = self.get_bounds(**kwarg)
                if minimum is not None:
                    constraints[f"min_{name}"] = minimum
                if maximum is not None:
                    constraints[f"max_{name}"] = maximum

        cols = st.columns([1] * 2, gap="large")



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

        self.low_margin_divider()
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

    def get_factor_constraints(self) -> Dict:
        params = {}
        c1, c2 = st.columns([4, 1])
        params["factors"] = c1.multiselect(
            label="Factor List",
            options=[
                factors.PriceMomentum1M.__name__,
                factors.PriceMomentum2M.__name__,
                factors.PriceMomentum3M.__name__,
                factors.PriceMomentum6M.__name__,
                factors.PriceMomentum9M.__name__,
                factors.PriceMomentum12M.__name__,
                factors.PriceMomentum24M.__name__,
                factors.PriceMomentum36M.__name__,
                factors.PriceMomentum6M1M.__name__,
                factors.PriceMomentum6M2M.__name__,
                factors.PriceMomentum9M1M.__name__,
                factors.PriceMomentum9M2M.__name__,
                factors.PriceMomentum12M1M.__name__,
                factors.PriceMomentum12M2M.__name__,
                factors.PriceMomentum24M1M.__name__,
                factors.PriceMomentum24M2M.__name__,
                factors.PriceMomentum36M1M.__name__,
                factors.PriceMomentum36M2M.__name__,
                factors.PriceVolatility1M.__name__,
                factors.PriceVolatility3M.__name__,
            ],
        )
        with c2:
            params["bounds"] = self.get_bounds(
                label="Factor Weight",
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                format_func=lambda x: f"{x:.0%}",
                help="Select range for your factor weight.",
            )
        return params

    def render(self):
        universe = self.get_universe(show=True)

        with st.form("AssetAllocationForm"):
            # Backtest Parameters
            bt_params = self.get_strategy_parameters()
            # Factor Implementatio
            factor_constraints = self.get_factor_constraints()
            bt_params["start"], bt_params["end"] = self.get_date_range()

            bt_params["allow_fractional_shares"] = self.get_allow_fractional_shares()

            # Asset Allocation Constraints
            with st.expander(label="Custom Constraints:"):
                st.subheader("Optimizer Constraint")
                optimizer_constraints = self.get_optimizer_constraints()
                self.low_margin_divider()
                st.subheader("Specific Constraint")
                specific_constraints = self.get_specific_constraints(universe=universe)



            submitted = st.form_submit_button(label="Backtest", type="primary")

            if submitted:
                prices = data.get_prices(tickers=universe.ticker.tolist())

                in_signiture = {
                    # "universe": universe,
                    "strategy": bt_params,
                    "constraints": {
                        "optimizer": optimizer_constraints,
                        "factor": factor_constraints,
                        "specific": specific_constraints,
                    },
                }

                no_duplicated_strategy = True
                for name, signiture in self.signiture.items():
                    if signiture == in_signiture:
                        st.warning(f"Exactly the same signiture with {name}")
                        self.no_duplicated_strategy = False

                if no_duplicated_strategy:
                    self.signiture[bt_params["name"]] = in_signiture
                    factor_bounds = factor_constraints["bounds"]

                    if (
                        factor_bounds == (None, None)
                        or not factor_constraints["factors"]
                    ):
                        factor_values = None
                    else:
                        with st.spinner("Loading Factor Data."):
                            factor_values = factors.multi.MultiFactors(
                                tickers=universe.ticker.tolist(),
                                factors=factor_constraints["factors"],
                            ).standard_percentile

                    with st.spinner(text="Backtesting in progress..."):
                        self.get_multistrategy().run(
                            **bt_params,
                            optimizer_constraints=optimizer_constraints,
                            specific_constraints=specific_constraints,
                            prices=prices,
                            factor_values=factor_values,
                            factor_bounds=factor_bounds,
                        )

        multistrategy = self.get_multistrategy()

        if multistrategy.strategies:
            st.button(
                label="Clear All Strategies",
                on_click=self.clear_strategy,
            )

            analytics = multistrategy.analytics
            st.dataframe(analytics.T, use_container_width=True)

            st.plotly_chart(
                components.charts.line(
                    data=self.get_multistrategy().values.resample("M").last(),
                    yaxis_title="NAV",
                    yaxis_tickformat="$,.0f",
                    hovertemplate="Date: %{x} - Value: %{y:,.0f}",
                    title="Strategy Performance",
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            for name, strategy in multistrategy.strategies.items():
                with st.expander(label=name, expanded=False):
                    st.button(
                        label=f"Delete {name}",
                        on_click=self.del_strategy,
                        kwargs={"name": name},
                    )
                    try:
                        st.json(self.signiture[name], expanded=False)
                    except KeyError:
                        st.warning("Signiture store not found.")

                    perf_tab, dd_tab, hw_tab, cw_tab = st.tabs(
                        ["Performance", "Drawdown", "Hist.Weights", "Curr.Weights"]
                    )

                    with perf_tab:
                        fig = components.charts.line(
                            strategy.value.to_frame(),
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
                        fig = components.charts.line(
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
                        fig = components.charts.line(
                            strategy.allocations,
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
                        fig = components.charts.pie(
                            strategy.allocations.iloc[-1].dropna(),
                            title="Strategy Current Weights",
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
