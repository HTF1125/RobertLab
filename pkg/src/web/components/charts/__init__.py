"""ROBERT"""

import pandas as pd
from plotly.subplots import make_subplots
from plotly.graph_objects import Scatter, Bar


def line(
    data: pd.DataFrame,
    title: str = "",
    xaxis_title: str = "Date",
    yaxis_title: str = "",
    hovemode: str = "x",
    hovertemplate: str = "Date: %{x}: %{y}}",
    xaxis_tickformat: str = "%Y-%m-%d",
    yaxis_tickformat: str = ".0%",
    legend_orientation: str = "v",
    legend_yanchor: str = "top",
    legend_xanchor: str = "left",
    legend_x: float = 0.0,
    legend_y: float = 1.1,
):
    fig = make_subplots(rows=1, cols=1)

    for col in data:
        trace = Scatter(
            x=data.index, y=data[col].values, name=col, hovertemplate=hovertemplate
        )
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        xaxis_tickformat=xaxis_tickformat,
        yaxis_title=yaxis_title,
        yaxis_tickformat=yaxis_tickformat,
        hovermode=hovemode,
        legend=dict(
            orientation=legend_orientation,
            yanchor=legend_yanchor,
            xanchor=legend_xanchor,
            y=legend_y,
            x=legend_x,
        ),
    )

    return fig


def bar(
    data: pd.DataFrame,
    title: str = "",
    xaxis_title: str = "Date",
    yaxis_title: str = "",
    hovemode: str = "x",
    hovertemplate: str = "Date: %{x}: %{y}}",
    xaxis_tickformat: str = "%Y-%m-%d",
    yaxis_tickformat: str = ".0%",
    legend_orientation: str = "v",
    legend_yanchor: str = "top",
    legend_xanchor: str = "left",
    legend_x: float = 0.0,
    legend_y: float = 1.1,
):
    fig = make_subplots(rows=1, cols=1)

    for col in data:
        trace = Bar(
            x=data.index, y=data[col].values, name=col, hovertemplate=hovertemplate
        )
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        xaxis_tickformat=xaxis_tickformat,
        yaxis_title=yaxis_title,
        yaxis_tickformat=yaxis_tickformat,
        hovermode=hovemode,
        legend=dict(
            orientation=legend_orientation,
            yanchor=legend_yanchor,
            xanchor=legend_xanchor,
            y=legend_y,
            x=legend_x,
        ),
    )

    return fig
