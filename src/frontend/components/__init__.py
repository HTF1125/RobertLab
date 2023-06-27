"""ROBERT"""
from typing import Optional
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.core.strategies import MultiStrategy
from src.core.benchmarks import Benchmark


def plot_multistrategy(multistrategy: MultiStrategy, allow_save: bool = False) -> None:
    for name, strategy in multistrategy.items():
        with st.expander(label=name, expanded=False):
            if allow_save:
                new_name = st.text_input(
                    label="Customize the strategy name",
                    key=f"custom name strategy {name}",
                    value=name,
                )

                # st.button(label="Save", on_click=save_strategy,)

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

                st.write(strategy.benchmark)
                if not strategy.benchmark is None:
                    fig.add_trace(
                        go.Scatter(
                            x=strategy.benchmark.performance.index,
                            y=strategy.benchmark.performance.values,
                            name="Benchmark",
                        )
                    )

                fig.update_layout(title="Performance", hovermode="x unified")
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
