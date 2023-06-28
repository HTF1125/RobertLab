"""ROBERT"""
from typing import Optional
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.core.strategies import MultiStrategy
from src.core.benchmarks import Benchmark


def delete_strategy(multistrategy: MultiStrategy, name: str):
    if multistrategy.delete(name):
        st.info(f"Delete strategy `{name}` successful.")
    else:
        st.warning("Delete Failed.")


def save_strategy(multistrategy: MultiStrategy, name: str, new_name: str):
    if multistrategy.save(name, new_name):
        st.info(f"Save strategy `{name}` successful.")
    else:
        st.warning("Save Failed.")


def plot_multistrategy(multistrategy: MultiStrategy, allow_save: bool = True) -> None:
    for name, strategy in multistrategy.items():
        with st.expander(label=name, expanded=False):

            if allow_save:
                new_name = st.text_input(
                    label="Customize the strategy name",
                    key=f"custom name strategy {name}",
                    value=name,
                )

                col1, col2 = st.columns([1, 1])
                col1.button(
                    label="Save",
                    key=f"{name}_save",
                    on_click=save_strategy,
                    kwargs={
                        "multistrategy": multistrategy,
                        "name": name,
                        "new_name": new_name,
                    },
                )

                col2.button(
                    label="Delete",
                    key=f"{name}_delete",
                    on_click=delete_strategy,
                    kwargs={"multistrategy": multistrategy, "name": name},
                )

            st.json(strategy.get_signature(), expanded=False)

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

                if not strategy.benchmark is None:
                    fig.add_trace(
                        go.Scatter(
                            x=strategy.benchmark.performance.index,
                            y=strategy.benchmark.performance.values,
                            name="Benchmark",
                        )
                    )

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

                if not strategy.benchmark is None:
                    fig.add_trace(
                        go.Scatter(
                            x=strategy.benchmark.drawdown.index,
                            y=strategy.benchmark.drawdown.values,
                            name="Benchmark",
                        )
                    )

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
