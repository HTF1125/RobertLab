import streamlit as st
from plotly.graph_objects import Scatter, Bar
from plotly.subplots import make_subplots
from .. import state


def main():
    if not state.strategy.get_backtestmanager().values.empty:
        analytics = state.strategy.get_backtestmanager().analytics
        st.write(analytics.T)

        fig = make_subplots(rows=1, cols=1)
        for name, strategy in state.strategy.get_backtestmanager().strategies.items():
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

    for name, strategy in state.strategy.get_backtestmanager().strategies.items():
        with st.expander(label=name, expanded=False):
            st.button(
                label="Delete",
                key=name,
                on_click=state.strategy.get_backtestmanager().drop_strategy,
                kwargs={"name": name},
            )

            st.json(strategy.analytics.to_dict(), expanded=False)

            performance, histo = st.tabs(
                [
                    "Performance",
                    "Hist.Allocations",
                ]
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
            fig.update_layout(
                title="Performance",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x",
                legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0),
                height=500,
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
            with performance:
                st.plotly_chart(
                    fig, use_container_width=True, config={"displayModeBar": False}
                )

            fig = make_subplots(rows=1, cols=1)

            for col in strategy.allocations.columns:
                fig.add_trace(
                    Bar(
                        x=strategy.allocations.index,
                        y=strategy.allocations[col],
                        name=col,
                    )
                )

            fig.update_layout(
                barmode="stack",
                title="Stacked Bar Chart",
                xaxis=dict(title="Date", tickformat="%Y-%m-%d"),
                yaxis=dict(title="Values", tickformat=".0%"),
                legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0),
                autosize=False,
                height=500,
            )
            with histo:
                st.plotly_chart(
                    fig, use_container_width=True, config={"displayModeBar": False}
                )
