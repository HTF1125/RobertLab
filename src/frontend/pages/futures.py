"""ROBERT"""
import streamlit as st
import plotly.graph_objects as go
from .base import BasePage


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

        universe = self.get_universe()
        prices = self.get_universe_prices(universe)
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
            title="From 52W High (y), Low (x)",
            hovermode="x unified",
            xaxis_tickformat=".1%",
            yaxis_tickformat=".1%",
            legend_orientation="h",
        )

        st.plotly_chart(fig, use_container_width=True)
