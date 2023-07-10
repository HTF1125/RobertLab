"""ROBERT"""
import streamlit as st
import plotly.graph_objects as go
from .base import BasePage
from .. import components
from ..components.strategy_button import strategy_button


class Futures(BasePage):
    """
    This sections holds all futures development prototypes.
    """

    def load_page(self):
        st.write(
            """
            Development Plan:

            <General>:
            1. Improve Code Efficiency.
            2. Add `__doc__` to classes.
            3. Remove data call in the frontend. (move all to backend data.)

            <Benchmarks>:
            1. Build class. (Done)
            2. Incorporate with portfolio optimizer. (Done)

            <Strategies>:
            1. Build Plot class. (Drop)


            """
        )

        self.chart_thought()

    def chart_thought(self):
        universe = components.single.get_universe()
        prices = universe.get_prices()
        minimum = prices / prices.rolling(252).min()
        maximum = prices / prices.rolling(252).max()
        fig = go.Figure()

        date = prices.index[-1]

        for ticker in prices.columns:
            fig.add_trace(
                go.Scatter(
                    x=[minimum.loc[date, ticker] - 1],
                    y=[maximum.loc[date, ticker] - 1],
                    name=ticker,
                    mode="markers+text",
                    text=ticker,
                    textposition="top center",
                    hovertemplate="AboveLow %{x}, BelowHigh: %{y}",
                )
            )

        fig.update_layout(
            xaxis_tickformat=".1%",
            yaxis_tickformat=".1%",
        )

        self.plotly(
            fig,
            title="From 52W High (y), Low (x)",
        )

        constraint = components.repeatable.Repeatable(prefix="g").fit()
        st.write(constraint)

        def run_component():
            value = strategy_button(
                key="strategy_button",
                buttons={
                    "create": False,
                    "delete": False,
                    "Save": False,
                },
            )
            return value

        def handle_event(value):
            st.write(value)


        handle_event(run_component())
