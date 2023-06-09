"""ROBERT"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.core import metrics, portfolios
from .base import BasePage
from .. import components


class CapitalMarket(BasePage):
    """
    Efficient Frontier
    """

    def load_page(self):
        universe = components.single.get_universe()

        with st.form("Efficient Frontier"):
            specific_constraints = components.single.get_asset_constraints(universe)
            submitted = st.form_submit_button("Run")

            if submitted:
                prices = universe.get_prices()
                with st.spinner(text="Loading Efficient Frontier for assets ..."):
                    ann_return = metrics.to_ann_return(prices=prices)
                    ann_volatility = metrics.to_ann_volatility(prices=prices)
                    ef = []
                    for ret in np.linspace(
                        start=ann_return.min(),
                        stop=ann_return.max(),
                        num=50,
                    ):
                        opt = portfolios.MinVolatility.from_prices(
                            prices=prices, min_return=ret, max_return=ret
                        ).set_specific_constraints(specific_constraints)
                        try:
                            opt.solve()
                            ef.append(
                                {
                                    "expected_return": opt.optimizer_metrics[
                                        "expected_return"
                                    ],
                                    "expected_volatility": opt.optimizer_metrics[
                                        "expected_volatility"
                                    ],
                                }
                            )
                        except ValueError:
                            pass

                    ef = pd.DataFrame(ef)

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=ef["expected_volatility"],
                            y=ef["expected_return"],
                            name="Efficient Frontier",
                            hovertemplate="Ann.Return: %{y}, Ann.Volatility: %{x}",
                        )
                    )
                    for ticker in ann_volatility.index:
                        fig.add_trace(
                            go.Scatter(
                                x=[ann_volatility.loc[ticker]],
                                y=[ann_return.loc[ticker]],
                                name=ticker,
                                mode="markers+text",
                                text=ticker,
                                textposition="top center",
                                hovertemplate="Ann.Return: %{y}, Ann.Volatility: %{x}",
                            )
                        )
                    fig.update_layout(
                        xaxis_title="Ann.Volatility",
                        yaxis_title="Ann.Return",
                        xaxis_tickformat=".2%",
                        yaxis_tickformat=".2%",
                        height=500,
                        hovermode="x",
                    )
                    self.plotly(fig)

                with st.spinner(text="Loading Efficient Frontier for allocations ..."):
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=ef["expected_volatility"],
                            y=ef["expected_return"],
                            name="Efficient Frontier",
                            hovertemplate="Ann.Return: %{y}, Ann.Volatility: %{x}",
                        )
                    )
                    weights = {}

                    markers = [
                        "star",
                        "square",
                        "diamond",
                        "cross",
                        "pentagon",
                        "hexagon",
                        "hourglass",
                        "star-square",
                        "star-square",
                    ]

                    for idx, optimizer_name in enumerate(portfolios.__all__):
                        optimizer = (
                            portfolios.get(optimizer_name)
                            .from_prices(prices=prices)
                            .set_specific_constraints(specific_constraints)
                        )
                        weights[optimizer_name] = optimizer.solve()

                        fig.add_trace(
                            go.Scatter(
                                x=[optimizer.optimizer_metrics["expected_volatility"]],
                                y=[optimizer.optimizer_metrics["expected_return"]],
                                name=optimizer_name,
                                text=optimizer_name,
                                mode="markers+text",
                                textposition="top center",
                                hovertemplate="Ann.Return: %{y}, Ann.Volatility: %{x}",
                                marker={"symbol": markers[idx], "size": 10},
                            )
                        )

                    fig.update_layout(
                        xaxis_title="Ann.Volatility",
                        yaxis_title="Ann.Return",
                        xaxis_tickformat=".2%",
                        yaxis_tickformat=".2%",
                        height=500,
                        hovermode="x",
                    )

                    self.plotly(fig)

                    for name, weight in weights.items():
                        with st.expander(label=name, expanded=False):
                            fig = go.Figure()

                            fig.add_trace(
                                go.Pie(
                                    labels=weight.index,
                                    values=weight.values,
                                )
                            )

                            self.plotly(fig)

