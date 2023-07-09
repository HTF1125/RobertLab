"""ROBERT"""
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from .base import BasePage
from .. import components


class Regime(BasePage):
    def load_page(self):
        with st.form(key="Regime state"):
            universe = components.get_universe()
            regime = components.get_regime()

            submitted = st.form_submit_button(label="Analyze")
            if submitted:
                prices = universe.get_prices()
                fwd_ret = prices.pct_change(21).shift(-21).dropna()
                fwd_ret["state"] = regime.states.loc[fwd_ret.index]

                fig = go.Figure()
                df = pd.get_dummies(regime.states)
                st.subheader("Regime state:")
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

                # Set the layout
                fig.update_layout(
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="State"),
                    hovermode="x",
                )
                self.plotly(fig)

                st.subheader("Performance by states")

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
                            hovertemplate="%{x} %{y:.2%}"
                        )
                    )

                # Set the layout
                fig.update_layout(hovermode="x")

                self.plotly(fig)
