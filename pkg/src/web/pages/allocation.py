"""ROBERT"""
import streamlit as st
from pkg.src.core import factors
from pkg.src.web import components, data
from pkg.src.core.strategies import MultiStrategy

def get_name() -> str:
    return str(
        st.text_input(
            label="Name",
            value=f"Strategy-{get_multistrategy().num_strategies}",
        )
    )
def get_multistrategy() -> MultiStrategy:
    # Initialization
    if "multi-strategy" not in st.session_state:
        st.session_state["multi-strategy"] = MultiStrategy()
    return st.session_state["multi-strategy"]


def get_portfolio_parameters():
    parameter_funcs = {
        "optimizer": components.get_optimizer,
        "benchmark": components.get_benchmark,
    }

    parameter_cols = st.columns([1] * len(parameter_funcs))

    parameters = {}

    for col, (name, func) in zip(parameter_cols, parameter_funcs.items()):
        with col:
            parameters[name] = func()
    return parameters


def get_strategy_parameters():
    parameter_funcs = {
        "start": components.get_start,
        "end": components.get_end,
        "frequency": components.get_frequency,
        "commission": components.get_commission,
    }

    parameter_cols = st.columns([1] * len(parameter_funcs))

    parameters = {}

    for col, (name, func) in zip(parameter_cols, parameter_funcs.items()):
        with col:
            parameters[name] = func()

    parameters.update(get_portfolio_parameters())

    parameters["allow_fractional_shares"] = components.get_allow_fractional_shares()

    return parameters


def main():
    strategy_signiture = {}

    universe = components.get_universe(show=True)

    with st.form("AssetAllocationForm"):
        name = get_name()
        # Backtest Parameters
        backtest_parameters = get_strategy_parameters()

        # Asset Allocation Constraints
        with st.expander(label="Asset Allocation Constraints:"):
            st.subheader("Optimizer Constraints")
            optimizer_constraints = components.get_optimizer_constraints()
            st.markdown("---")
            st.subheader("Specific Constraints")
            specific_constraints = components.get_specific_constraints(
                universe=universe
            )
            st.markdown("---")

        # Factor Implementation
        with st.expander(label="Factor Implementation:"):
            factor_constraints = components.get_factor_constraints()

        submitted = st.form_submit_button(label="Backtest", type="primary")

        if submitted:
            prices = data.get_prices(tickers=universe.ticker.tolist())

            strategy_signiture[backtest_parameters["name"]] = {
                "strategy": backtest_parameters,
                "constraints": {
                    "optimizer": optimizer_constraints,
                    "factor": factor_constraints,
                    "specific": specific_constraints,
                },
            }

            if factor_constraints["factors"]:
                with st.spinner("Loading Factor Data."):
                    factor_values = factors.multi.MultiFactors(
                        tickers=universe.ticker.tolist(),
                        factors=factor_constraints["factors"],
                    ).standard_percentile
            else:
                factor_values = None

            with st.spinner(text="Backtesting in progress..."):
                get_multistrategy().run(
                    name=name,
                    **backtest_parameters,
                    optimizer_constraints=optimizer_constraints,
                    specific_constraints=specific_constraints,
                    prices=prices,
                    factor_values=factor_values,
                    factor_bounds=factor_constraints["bounds"],
                )
                setattr(
                    get_multistrategy().strategies[backtest_parameters["name"]],
                    "signiture",
                    strategy_signiture,
                )

    st.button(
        label="Clear All Strategies",
        on_click=get_multistrategy().reset_strategies,
    )

    multistrategy = get_multistrategy()

    if multistrategy.strategies:
        analytics = multistrategy.analytics
        st.dataframe(analytics.T, use_container_width=True)

        st.plotly_chart(
            components.charts.line(
                data=get_multistrategy().values.resample("M").last(),
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
                st.json(getattr(strategy, "signiture"), expanded=False)

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
                        fig, use_container_width=True, config={"displayModeBar": False}
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
                        fig, use_container_width=True, config={"displayModeBar": False}
                    )
                with hw_tab:
                    fig = components.charts.line(
                        strategy.allocations,
                        xaxis_tickformat="%Y-%m-%d",
                        xaxis_title="Date",
                        yaxis_title="Weights",
                        yaxis_tickformat=".0%",
                        hovertemplate="Date: %{x} - Value: %{y:.2%}",
                        title="Strategy Performance",
                        stackgroup="stack",
                    )
                    st.plotly_chart(
                        fig, use_container_width=True, config={"displayModeBar": False}
                    )

                with cw_tab:
                    fig = components.charts.pie(strategy.allocations.iloc[-1].dropna())
                    st.plotly_chart(
                        fig, use_container_width=True, config={"displayModeBar": False}
                    )
