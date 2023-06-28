"""ROBERT"""

import streamlit as st
import plotly.graph_objects as go
from src.core import MultiStrategy
from .base import BasePage
from ..components import plot_multistrategy


class Dashboard(BasePage):
    """Dashboard"""

    def load_page(self):
        st.info("Welcome to Robert's Dashboard.")

        multistrategy = MultiStrategy()
        fig = go.Figure()

        with st.spinner("load strategies..."):
            multistrategy.from_files()
        st.write(multistrategy.analytics.T)

        show_alpha = st.checkbox(label="show alpha")
        for name, strategy in multistrategy.items():

            if show_alpha:
                performance = strategy.performance_alpha
            else:
                performance = strategy.performance
            fig.add_trace(
                go.Scatter(
                    x=performance.index,
                    y=performance.values,
                    name=name,
                )
            )

        fig.update_layout(
            title="Alpha Performance", legend_orientation="h", hovermode="x unified"
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        plot_multistrategy(multistrategy, allow_save=False)
