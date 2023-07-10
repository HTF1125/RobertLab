"""ROBERT"""
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.core import metrics
from .base import BasePage
from .. import components


class MarketRegime(BasePage):

    """
    In <MarketRegime>, you make find some pre-defined regimes,
    """

    def load_page(self):
        with st.form(key="Regime state"):
            col1, col2, col3 = st.columns(3)
            with col1:
                universe = components.single.get_universe()
            with col2:
                regime = components.single.get_regime()
            with col3:
                periods = components.single.get_periods()

            submitted = st.form_submit_button(label="Analyze")
            if submitted:
                prices = universe.get_prices()
                fwd_ret = metrics.to_log_return(
                    prices, periods=periods, forward=True
                ) * (252 / periods)
                fwd_ret["state"] = regime.get_states().loc[fwd_ret.index]

                fig = go.Figure()
                df = pd.get_dummies(regime.get_states())
                for column in df.columns:
                    state = column
                    y = df[column]
                    fig.add_trace(
                        go.Scatter(
                            x=y.index,
                            y=y.values,
                            mode="lines",
                            fill="tozeroy",
                            name=state,
                        )
                    )

                self.plotly(fig, title="Regime Map")


                df = fwd_ret.groupby(by="state").mean()

                fig = go.Figure()

                for column in df.columns:
                    state = column
                    y = df[column]
                    labels = [f"{value:.2%}" for value in y.values]

                    fig.add_trace(
                        go.Bar(
                            x=y.index,
                            y=y.values,
                            text=labels,
                            textposition="auto",
                            name=state,
                            hovertemplate="%{y:.2%}",
                        )
                    )
                fig.update_layout(yaxis_tickformat=".2%")

                self.plotly(fig, title=f"Mean annualized performance by state (forward {periods})")
