"""ROBERT"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pkg.src.core import metrics, portfolios
from pkg.src.web import components
from pkg.src.web import data
from pkg.src.web.components.charts import pie


def main():
    with st.form("Efficient Frontier"):
        universe = components.get_universe(show=True)

        start, end = components.get_date_range()

        tickers = universe.ticker.tolist()

        submitted = st.form_submit_button("Run")

        if submitted:
            prices = (
                data.get_prices(tickers=tickers).ffill().loc[start:end].dropna(axis=1)
            )

            ann_return = metrics.to_ann_return(prices=prices)
            ann_volatility = metrics.to_ann_volatility(prices=prices)

            ef = []

            for ret in np.linspace(
                start=ann_return.min(),
                stop=ann_return.max(),
                num=50,
            ):
                opt = portfolios.MinVolatility.from_prices(prices=prices).set_bounds(
                    port_return=(ret, ret)
                )
                try:
                    opt.solve()
                    ef.append(
                        {
                            "expected_return": opt.exp["expected_return"],
                            "expected_volatility": opt.exp["expected_volatility"],
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
                    name="Eff.Frontier",
                    hovertemplate="Ann.Return: %{y}, Ann.Volatility: %{x}",
                )
            )

            ef_points = [
                dict(
                    optimizer=portfolios.MaxSharpe,
                    marker=dict(symbol="star", size=10, color="cyan"),
                ),
                dict(
                    optimizer=portfolios.MinVolatility,
                    marker=dict(symbol="square", size=10, color="cyan"),
                ),
                dict(
                    optimizer=portfolios.MinCorrelation,
                    marker=dict(symbol="diamond", size=10, color="cyan"),
                ),
                dict(
                    optimizer=portfolios.RiskParity,
                    marker=dict(symbol="cross", size=10, color="cyan"),
                ),
                dict(
                    optimizer=portfolios.HRP,
                    marker=dict(symbol="pentagon", size=10, color="cyan"),
                ),
                dict(
                    optimizer=portfolios.HERC,
                    marker=dict(symbol="hexagon", size=10, color="cyan"),
                ),
                dict(
                    optimizer=portfolios.MaxReturn,
                    marker=dict(symbol="hourglass", size=10, color="cyan"),
                ),
                dict(
                    optimizer=portfolios.EqualWeight,
                    marker=dict(symbol="star-square", size=10, color="cyan"),
                ),
                dict(
                    optimizer=portfolios.InverseVariance,
                    marker=dict(symbol="star-square", size=10, color="cyan"),
                ),
            ]

            weights = {}

            for ef_point in ef_points:
                optimizer = ef_point["optimizer"]
                marker = ef_point["marker"]
                opt = optimizer.from_prices(prices=prices).set_bounds()
                name = opt.__class__.__name__
                weights[name] = opt.solve()
                fig.add_trace(
                    go.Scatter(
                        x=[opt.exp["expected_volatility"]],
                        y=[opt.exp["expected_return"]],
                        name=name,
                        text=name,
                        mode="markers+text",
                        textposition="top center",
                        hovertemplate="Ann.Return: %{y}, Ann.Volatility: %{x}",
                        marker=marker,
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

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

            for name, weight in weights.items():
                with st.expander(label=name, expanded=False):
                    st.plotly_chart(pie(data=weight), use_container_width=True)
