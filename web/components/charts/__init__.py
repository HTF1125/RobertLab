

from plotly.subplots import make_subplots
from plotly.graph_objects import Scatter


def line(data):

    fig = make_subplots(rows=1, cols=1)

    for col in data:

        trace = Scatter(
            x = data.index,
            y = data[col].values,
            name = col,
            hovertemplate="Data: %{x}: %{y}"
        )
        fig.add_trace(trace)

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
            tickprefix="", # Add a currency symbol to the y-axis tick labels
            ticksuffix="%"
        ),
    )
    return fig