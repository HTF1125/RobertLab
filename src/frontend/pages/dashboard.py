"""ROBERT"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from .base import BasePage
from ..components import plot_multistrategy
from .. import session

class Dashboard(BasePage):
    """
    In <Dashboard>, you will find a selection of pre-defined strategies
    that are readily available.\n
    However, if you wish to explore and experiment with new strategies, you can
    do so using the <MultiStrategy> feature.\n
    The <MultiStrategy> allows you to test and refine your custom strategies.
    Once you have developed a strategy that meets your requirements, you can
    save it, and it will be added to the dashboard page for easy access.
    """

    def load_page(self):
        st.button(
            label="➕ New Strategy",
            on_click=session.setPage,
            kwargs={"page": "MultiStrategy", "rerun": False},
        )
        with st.spinner(text="Loading pre-defined strategies..."):
            multistrategy = session.getPreStrategy()

        if multistrategy:
            fig = go.Figure()
            for name, strategy in multistrategy.items():
                performance = strategy.performance / strategy.principal - 1
                num_points = len(performance)
                indices = np.linspace(0, num_points - 1, 100, dtype=int)
                performance = performance.iloc[indices]
                fig.add_trace(
                    go.Scatter(
                        x=performance.index,
                        y=performance.values,
                        name=name,
                        showlegend=True,
                    )
                )
            fig.update_layout(
                legend_orientation="h",
                hovermode="x unified",
                yaxis_tickformat=".0%",
            )
            self.plotly(fig, title="Strategy Performance")
            self.h3("Strategy Metrics")
            st.table(multistrategy.analytics.T)
            self.h3(text="Strategy Analysis:")
            plot_multistrategy(multistrategy, allow_save=False)

        else:

            st.info("There is not pre-defined strategies, click to add <strategy>")