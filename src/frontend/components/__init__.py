"""ROBERT"""
from typing import Optional, Tuple, List, Dict, Callable, Any
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.core.strategies.multi import MultiStrategy
from src.core import universes, benchmarks, portfolios, factors, regimes
from .session import Session
from .constraint_set import PortfolioModelSet

from ..components.future import StrategyParameters, StrategyConstraint, get_universe, get_regime, get_strategy





def get_factor() -> None:
    st.session_state["factors"] = tuple(
        st.multiselect(
            label="Factor List",
            options=factors.__all__,
        )
    )


def get_allow_fractional_shares() -> bool:
    return st.checkbox(
        label="Fractional Shares",
        value=False,
        help="Allow Fractional Shares Investing.",
    )


def get_frequency() -> str:
    options = ["D", "M", "Q", "Y"]
    return str(
        st.selectbox(
            label="Freq",
            options=options,
            index=options.index(
                st.session_state.get("frequency", "M"),
            ),
            help="Select strategy's rebalancing frequency.",
        )
    )


def get_inception() -> pd.Timestamp:
    return pd.Timestamp(
        str(
            st.date_input(
                label="Incep",
                value=pd.Timestamp("2003-01-01"),
            )
        )
    )


def get_commission() -> int:
    return int(
        st.number_input(
            label="Comm",
            min_value=0,
            max_value=100,
            step=10,
            value=st.session_state.get("commission", 10),
            help="Select strategy's trading commission in basis points.",
        )
    )


def get_min_window() -> int:
    return int(
        st.number_input(
            label="Win",
            min_value=2,
            max_value=1500,
            step=100,
            value=st.session_state.get("min_window", 252),
            help="Minimum window of price data required.",
        )
    )


def get_portfolio():
    return portfolios.get(
        str(
            st.selectbox(
                label="Port",
                options=portfolios.__all__,
                help="Select strategy's rebalancing frequency.",
            )
        )
    )



def get_dates(
    start: str = "2010-01-01",
    end: str = str(date.today()),
) -> Tuple[str, str]:
    DATEFORMAT = "%Y-%m-%d"
    dates = pd.date_range(start=start, end=end, freq="M")
    dates = [date.strftime(DATEFORMAT) for date in dates]

    selected_start, selected_end = st.select_slider(
        "Date Range",
        options=dates,
        value=(dates[0], dates[-1]),
    )
    if selected_start >= selected_end:
        st.error("Error: The start date must be before the end date.")
    return selected_start, selected_end


def get_bounds(
    label: str,
    min_value: float = 0,
    max_value: float = 100,
    step: float = 4,
    format_func: Callable[[Any], Any] = str,
    help: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float]]:
    bounds = st.select_slider(
        label=label,
        options=np.arange(min_value, max_value + step, step),
        value=(min_value, max_value),
        format_func=format_func,
        help=help,
    )
    assert isinstance(bounds, tuple)
    low, high = bounds

    return (
        low if low != min_value else None,
        high if high != max_value else None,
    )


def get_specific_constraints(
    universe: universes.Universe, num_columns: int = 5
) -> List[Dict]:
    constraints = []
    universe_df = pd.DataFrame(universe.ASSETS)
    asset_classes = universe_df["assetclass"].unique()
    final_num_columns = min(num_columns, len(asset_classes))
    cols = st.columns([1] * final_num_columns, gap="large")
    for idx, asset_class in enumerate(asset_classes):
        with cols[idx % num_columns]:
            bounds = get_bounds(
                label=asset_class,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format_func=lambda x: f"{x:.0%}",
            )
            if bounds == (None, None):
                continue
            constraint = {
                "assets": universe_df[
                    universe_df["assetclass"] == asset_class
                ].ticker.to_list(),
                "bounds": bounds,
            }
        constraints.append(constraint)
    with st.expander("Specific Asset Constrints:"):
        final_num_columns = min(num_columns, len(universe_df))
        cols = st.columns([1] * final_num_columns, gap="large")
        for idx, asset in enumerate(universe_df.to_dict("records")):
            ticker = asset["ticker"]
            name = asset["name"]
            with cols[idx % num_columns]:
                bounds = get_bounds(
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


def delete_strategy(multistrategy: MultiStrategy, name: str):
    if multistrategy.delete(name):
        st.info(f"Delete strategy `{name}` successful.")
        st.experimental_rerun()
    else:
        st.warning("Delete Failed.")


def save_strategy(multistrategy: MultiStrategy, name: str, new_name: str):
    if multistrategy.save(name, new_name):
        st.info(f"Save strategy `{new_name}` successful.")
        st.experimental_rerun()
    else:
        st.warning("Save Failed.")


def plot_multistrategy(multistrategy: MultiStrategy, allow_save: bool = False) -> None:
    for name, strategy in multistrategy.items():
        with st.expander(label=name, expanded=False):
            if allow_save:
                with st.form(f"{name}"):
                    new_name = st.text_input(
                        label="Customize the strategy name",
                        key=f"custom name strategy {name}",
                        value=name,
                    )
                    action = st.radio(
                        label="Action",
                        options=["Save", "Delete"],
                        horizontal=True,
                        label_visibility="collapsed",
                    )
                    submitted = st.form_submit_button(label="submit")
                    if submitted:
                        if action == "Save":
                            save_strategy(multistrategy, name, new_name)
                        elif action == "Delete":
                            delete_strategy(multistrategy, name)

                signature = strategy.get_signature()
                del signature["book"]
                st.json(signature, expanded=False)

            (
                performance_tab,
                drawdown_tab,
                hist_allocations_tab,
                curr_allocations_tab,
            ) = st.tabs(
                [
                    "Performance",
                    "Drawdown",
                    "Hist. Allocations",
                    "Curr. Allocations",
                ]
            )

            with performance_tab:
                performance = strategy.performance

                fig = go.Figure().add_trace(
                    go.Scatter(
                        x=performance.index,
                        y=performance.values,
                        name="Performance",
                    )
                )

                # if not strategy.benchmark is None:
                #     bmP = strategy.benchmark.performance.reindex(performance.index).ffill()
                #     bmP = bmP / bmP.iloc[0] * performance.iloc[0]
                #     fig.add_trace(
                #         go.Scatter(
                #             x=bmP.index,
                #             y=bmP.values,
                #             name="Benchmark",
                #         )
                #     )

                fig.update_layout(
                    title="Performance", hovermode="x unified", legend_orientation="h"
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            with drawdown_tab:
                drawdown = strategy.drawdown
                fig = go.Figure().add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown.values,
                        name="Performance",
                    )
                )

                # if not strategy.benchmark is None:
                #     fig.add_trace(
                #         go.Scatter(
                #             x=strategy.benchmark.drawdown.index,
                #             y=strategy.benchmark.drawdown.values,
                #             name="Benchmark",
                #         )
                #     )

                fig.update_layout(
                    title="Performance", hovermode="x unified", legend_orientation="h"
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            allocations = strategy.allocations

            with hist_allocations_tab:
                fig = go.Figure()

                for asset in allocations:
                    fig.add_trace(
                        go.Scatter(
                            x=allocations.index,
                            y=allocations[asset].values,
                            name=asset,
                            stackgroup="one",
                        )
                    )

                fig.update_layout(
                    xaxis_tickformat="%Y-%m-%d",
                    xaxis_title="Date",
                    yaxis_title="Weights",
                    yaxis_tickformat=".0%",
                    title="Strategy Historical Weights",
                    hovermode="x unified",
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            with curr_allocations_tab:
                curr_allocations = allocations.iloc[-1].dropna()
                curr_allocations = curr_allocations[curr_allocations != 0.0]
                fig = (
                    go.Figure()
                    .add_trace(
                        go.Pie(
                            labels=curr_allocations.index,
                            values=curr_allocations.values,
                        )
                    )
                    .update_layout(
                        hovermode="x unified",
                    )
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
