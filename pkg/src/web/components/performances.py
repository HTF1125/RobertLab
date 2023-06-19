import streamlit as st
from plotly.graph_objects import Scatter, Bar, Pie
from plotly.subplots import make_subplots
from pkg.src.core.strategies import MultiStrategy





def main(multistrategy: MultiStrategy):


    for name, strategy in multistrategy.strategies.items():
        with st.expander(label=name, expanded=False):
            st.json(getattr(strategy, "signiture"))

            st.button(
                label="Delete",
                key=name,
                on_click=multistrategy.drop_strategy,
                kwargs={"name": name},
            )

            st.json(strategy.analytics.to_dict(), expanded=False)

            performance, histo, drawdown = st.tabs(
                [
                    "Performance",
                    "Hist.Allocations",
                    "Drawdown",
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
            fig.add_trace(price_trace, row=1, col=1)
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

            fig = make_subplots(rows=1, cols=1)

            fig.add_trace(
                Scatter(
                    x=strategy.drawdown.index,
                    y=strategy.drawdown,
                    mode="lines",
                    name="Drawdown",
                )
            )

            with drawdown:
                st.plotly_chart(
                    fig, use_container_width=True, config={"displayModeBar": False}
                )
