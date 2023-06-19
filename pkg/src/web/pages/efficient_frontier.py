"""ROBERT"""
import numpy as np
import pandas as pd
import streamlit as st
from pkg.src.web import components
from pkg.src.web import data
from pkg.src.core import metrics, portfolios
import plotly.graph_objects as go



def main():
    universe = components.get_universe(show=True)

    start, end = components.get_date_range()

    tickers = universe.ticker.tolist()
    prices = data.get_prices(tickers=tickers).ffill().loc[start:end].dropna(axis=1)

    ann_return = metrics.to_ann_return(prices=prices)
    ann_volatility = metrics.to_ann_volatility(prices=prices)

    ef = []

    for ret in np.arange(
        max(round(ann_return.min(), 2) + 0.01, 0), round(ann_return.max(), 2), 0.005
    ):
        try:
            opt = portfolios.MinVolatility.from_prices(prices=prices).set_bounds(
                port_return=(ret, ret)
            )
            opt()
            ef.append(
                {
                    "expected_return": opt.exp["expected_return"],
                    "expected_volatility": opt.exp["expected_volatility"],
                }
            )
        except:
            continue

    ef = pd.DataFrame(ef)

    fig = go.Figure()

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

    fig.add_trace(
        go.Scatter(
            x=ef["expected_volatility"], y=ef["expected_return"], name="Eff.Frontier"
        )
    )

    fig.update_layout(
        xaxis_title="Ann.Volatility",
        yaxis_title="Ann.Return",
        xaxis_tickformat=".2%",
        yaxis_tickformat=".2%",
        height=500,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )
