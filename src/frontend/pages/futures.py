"""ROBERT"""
import streamlit as st
from .base import BasePage


class Futures(BasePage):
    """
    This sections holds all futures development prototypes.
    """

    def load_page(self):
        st.write(
            """
            Development Plan:

            `Benchmarks`:
            1. Build class.
            2. Incorporate with portfolio optimizer.

            `Strategies`:
            1. Build Plot class.
            2. Make in class plot functions with Plot class.

            """
        )

        self.chart_thought()

    def chart_thought(self):
        import plotly.graph_objects as go

        self.get_universe()
        prices = self.get_universe_prices()
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
            title="From the 52w High&Low",
            hovermode="x unified",
            xaxis_title="Above 52w Low",
            yaxis_title="Below 52w Hight",
            xaxis_tickformat=".2%",
            yaxis_tickformat=".2%",

        )

        st.plotly_chart(fig, use_container_width=True)
