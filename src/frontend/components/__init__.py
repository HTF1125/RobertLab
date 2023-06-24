"""ROBERT"""
from typing import Optional
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


class Plot:
    def __init__(self) -> None:
        self.fig = go.Figure()

    def Bar(
        self,
        data: pd.DataFrame,
        hovertemplate: Optional[str] = None,
        stackgroup: Optional[str] = None,
    ) -> "Plot":
        for col in data:
            trace = go.Bar(
                x=data.index,
                y=data[col].values,
                name=col,
                hovertemplate=hovertemplate,
                stackgroup=stackgroup,
            )
            self.fig.add_trace(trace)

        return self

    def Pie(
        self,
        data: pd.DataFrame,
        hovertemplate: Optional[str] = None,
        stackgroup: Optional[str] = None,
    ) -> "Plot":
        for col in data:
            trace = go.Pie(
                x=data.index,
                y=data[col].values,
                name=col,
                hovertemplate=hovertemplate,
                stackgroup=stackgroup,
            )
            self.fig.add_trace(trace)

        return self

    def Line(
        self,
        data: pd.DataFrame,
        hovertemplate: Optional[str] = None,
        stackgroup: Optional[str] = None,
    ) -> "Plot":
        for col in data:
            trace = go.Scatter(
                x=data.index,
                y=data[col].values,
                name=col,
                hovertemplate=hovertemplate,
                stackgroup=stackgroup,
            )
            self.fig.add_trace(trace)

        return self

    def update_layout(
        self,
        title: Optional[str] = None,
        xaxis_title: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        hovermode: Optional[str] = None,
        xaxis_tickformat: Optional[str] = None,
        yaxis_tickformat: Optional[str] = None,
        legend_orientation: Optional[str] = None,
        legend_yanchor: Optional[str] = None,
        legend_xanchor: Optional[str] = None,
        legend_x: Optional[float] = None,
        legend_y: Optional[float] = None,
        **kwargs,
    ) -> "Plot":
        self.fig.update_layout(
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
            **kwargs
        )
        return self

    def load_streamlit(self, **kwargs) -> None:
        st.plotly_chart(
            figure_or_data=self.fig,
            use_container_width=True,
            config={"displayModeBar": False},
            **kwargs
        )
