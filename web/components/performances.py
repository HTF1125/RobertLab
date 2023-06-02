import streamlit as st
from plotly.graph_objects import Scatter
from plotly.subplots import make_subplots
from ..state import get_backtestmanager


def main():

    if not get_backtestmanager().values.empty:

        st.write(get_backtestmanager().analytics.T)

        fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
        )
        for name, strategy in get_backtestmanager().strategies.items():
            # Add line chart for prices to the first subplot
            val = strategy.value.resample("M").last()
            price_trace = Scatter(
                x=val.index,
                y=val.values,
                name=name,
                hovertemplate="Date: %{x} Price: %{y}",
            )
            fig.add_trace(price_trace)

        fig.update_layout(
            title="Performance",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x",
            legend=dict(orientation="v", yanchor="top", y=1.1, xanchor="left", x=0),
        )

        fig.update_layout(
            xaxis=dict(
                title="Date", tickformat="%Y-%m-%d"
            ),  # Customize the date format
            yaxis=dict(
                title="Price",
                tickprefix="$",  # Add a currency symbol to the y-axis tick labels
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    for name, strategy in get_backtestmanager().strategies.items():
        with st.expander(label=name, expanded=False):
            st.button(
                label="Delete",
                key=name,
                on_click=get_backtestmanager().drop_strategy,
                kwargs={"name": name},
            )
            fig = make_subplots(
                rows=1,
                cols=1,
                shared_xaxes=True,
            )
            price_trace = Scatter(
                x=strategy.value.index,
                y=strategy.value.values,
                name="Strategy",
                hovertemplate="Date: %{x}<br>Price: %{y}",
            )
            price_bm_trace = Scatter(
                x=strategy.prices_bm.index,
                y=strategy.prices_bm.values,
                name="Benchmark",
                hovertemplate="Date: %{x}<br>Price: %{y}",
            )
            fig.add_trace(price_trace, row=1, col=1)
            fig.add_trace(price_bm_trace, row=1, col=1)
            # strategy_dd = metrics.to_drawdown(strategy.value)
            # benchmark_dd = metrics.to_drawdown(strategy.prices_bm)
            # price_trace = Scatter(
            #     x=strategy_dd.index,
            #     y=strategy_dd.values,
            #     name="Strategy",
            #     hovertemplate="Date: %{x}<br>Price: %{y}",
            # )
            # price_bm_trace = Scatter(
            #     x=benchmark_dd.index,
            #     y=benchmark_dd.values,
            #     name="Benchmark",
            #     hovertemplate="Date: %{x}<br>Price: %{y}",
            # )

            # fig.add_trace(strategy_dd, row=2, col=1)
            # fig.add_trace(benchmark_dd, row=2, col=1)
            fig.update_layout(
                title="Performance",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x",
                legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0),
            )

            fig.update_layout(
                xaxis=dict(
                    title="Date", tickformat="%Y-%m-%d"
                ),  # Customize the date format
                yaxis=dict(
                    title="Price",
                    tickprefix="$",  # Add a currency symbol to the y-axis tick labels
                ),
            )

            st.plotly_chart(fig, use_container_width=True)
