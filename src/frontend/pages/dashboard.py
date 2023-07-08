"""ROBERT"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from .base import BasePage
from ..components import get_strategy, plot_multistrategy


class Dashboard(BasePage):
    """Dashboard"""

    def load_page(self):
        st.info("Welcome to Robert's Dashboard.")

        multistrategy = get_strategy()

        self.subheader("Strategy Metrics")
        self.divider()
        st.table(multistrategy.analytics.T)


        if multistrategy:
            fig = go.Figure()

            for name, strategy in multistrategy.items():
                performance = strategy.performance / strategy.initial_investment - 1
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


            self.subheader("Strategy Performance")
            self.divider()
            self.plotly(fig)


            # plot_multistrategy(multistrategy, allow_save=False)
