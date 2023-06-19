"""ROBERT"""
from typing import Optional, List, Union
import pandas as pd
from plotly.subplots import make_subplots
from plotly.graph_objects import Scatter, Bar
import plotly.graph_objects as go


def line(
    data: pd.DataFrame,
    title: str = "",
    xaxis_title: str = "Date",
    yaxis_title: str = "",
    hovermode: str = "x",
    hovertemplate: str = "Date: %{x}: %{y}}",
    xaxis_tickformat: str = "%Y-%m-%d",
    yaxis_tickformat: str = ".0%",
    legend_orientation: Optional[str] = None,
    legend_yanchor: Optional[str] = None,
    legend_xanchor: Optional[str] = None,
    legend_x: Optional[float] = None,
    legend_y: Optional[float] = None,
    stackgroup: Optional[str] = None
):
    fig = make_subplots(rows=1, cols=1)

    for col in data:
        trace = Scatter(
            x=data.index, y=data[col].values, name=col, hovertemplate=hovertemplate, stackgroup=stackgroup
        )
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        xaxis_tickformat=xaxis_tickformat,
        yaxis_title=yaxis_title,
        yaxis_tickformat=yaxis_tickformat,
        hovermode=hovermode,
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
    legend_orientation: Optional[str] = None,
    legend_yanchor: Optional[str] = None,
    legend_xanchor: Optional[str] = None,
    legend_x: Optional[float] = None,
    legend_y: Optional[float] = None,
    barmode="stack",
):
    fig = make_subplots(rows=1, cols=1)

    for col in data:
        trace = Bar(
            x=data.index, y=data[col].values, name=col, hovertemplate=hovertemplate
        )
        fig.add_trace(trace)

    fig.update_layout(
        barmode=barmode,
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


def pie(
    # *data : List[Union[pd.Series, pd.DataFrame]],
    data: pd.Series,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    hovemode: Optional[str] = None,
    hovertemplate: Optional[str] = None,
    xaxis_tickformat: Optional[str] = None,
    yaxis_tickformat: Optional[str] = None,
    legend_orientation: Optional[str] = None,
    legend_yanchor: Optional[str] = None,
    legend_xanchor: Optional[str] = None,
    legend_x: Optional[float] = None,
    legend_y: Optional[float] = None,
):
    fig = go.Figure()

    fig.add_trace(go.Pie(labels=data.index, values=data.values))
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
