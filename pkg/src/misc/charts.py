"""ROBERT"""

import pandas as pd
from plotly.subplots import make_subplots
from plotly.graph_objects import Scatter, Bar


def set_layout(
    fig,
    title: str = "",
    xaxis_title: str = "Date",
    yaxis_title: str = "",
    hovemode: str = "x",
    hovertemplate: str = "Date: %{x}: %{y}",
    xaxis_tickformat: str = "%Y-%m-%d",
    yaxis_tickformat: str = ".0%",
    legend_orientation: str = "v",
    legend_yanchor: str = "top",
    legend_xanchor: str = "left",
    legend_x: float = 0.0,
    legend_y: float = 1.1,
):

    fig.update_traces(
        hovertemplate=hovertemplate,
    )

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


def create_lineplot(
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

    for col in data.columns:
        fig.add_trace(Scatter(x=data.index, y=data[col], name=col))

    set_layout(
        fig,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovemode=hovemode,
        hovertemplate=hovertemplate,
        xaxis_tickformat=xaxis_tickformat,
        yaxis_tickformat=yaxis_tickformat,
        legend_orientation=legend_orientation,
        legend_yanchor=legend_yanchor,
        legend_xanchor=legend_xanchor,
        legend_x=legend_x,
        legend_y=legend_y,
    )

    return fig


def create_barplot(
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

    for col in data.columns:
        fig.add_trace(
            Bar(x=data.index, y=data[col], name=col)
        )

    set_layout(
        fig,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovemode=hovemode,
        # hovertemplate=hovertemplate,
        xaxis_tickformat=xaxis_tickformat,
        yaxis_tickformat=yaxis_tickformat,
        legend_orientation=legend_orientation,
        legend_yanchor=legend_yanchor,
        legend_xanchor=legend_xanchor,
        legend_x=legend_x,
        legend_y=legend_y,
    )

    return fig
