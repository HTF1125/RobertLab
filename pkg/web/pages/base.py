"""ROBERT"""
import os
import re
from typing import Tuple, Optional, Callable, Any
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pkg.src import data
from pkg import web


class BasePage:
    def __init__(self) -> None:
        self.load_local_css()
        self.load_font_awesome_style()
        self.load_social_media_badges()
        self.load_page_header()
        self.render()

    def render(self):
        st.warning("The Page is under construction...")

    @staticmethod
    def load_local_css():
        file = os.path.join(os.path.dirname(web.__file__), "css", "base.css")
        with open(file=file, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    @staticmethod
    def load_font_awesome_style():
        url = (
            "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
        )
        st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

    @staticmethod
    def load_social_media_badges():
        css_example = """
            <div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%);">
                <a href="https://github.com/htf1125" target="_blank" style="text-decoration: none;">
                    <i class="fa-brands fa-github fa-lg" style="margin-right: 5px; margin-left: 5px; vertical-align: middle; color: #808080;"></i>
                </a>
                <a href="https://www.linkedin.com/in/htf1125" target="_blank" style="text-decoration: none;">
                    <i class="fa-brands fa-linkedin fa-lg" style="margin-right: 5px; margin-left: 5px; vertical-align: middle; color: #808080;"></i>
                </a>
            </div>
        """
        st.markdown(css_example, unsafe_allow_html=True)

    def load_page_header(self):
        st.subheader(self.add_spaces_to_pascal_case(self.__class__.__name__))
        self.low_margin_divider()

    @staticmethod
    def low_margin_divider():
        st.markdown(
            '<hr style="margin-top: 0px; margin-bottom: 5px;">', unsafe_allow_html=True
        )

    @staticmethod
    def add_spaces_to_pascal_case(string):
        # Use regular expressions to add spaces before capital letters
        spaced_string = re.sub(r"(?<!^)(?=[A-Z])", " ", string)
        return spaced_string

    @staticmethod
    def get_universe(show: bool = False) -> pd.DataFrame:
        file = os.path.join(os.path.dirname(data.__file__), "universe.csv")
        file_load = pd.read_csv(file)
        selected = st.selectbox(
            label="Select Investment Universe",
            options=file_load.universe.unique(),
            help="Select strategy's investment universe.",
        )
        universe = file_load[file_load.universe == selected]

        if show:
            with st.expander(label="See universe details:"):
                st.json(universe.to_dict("records"))

        return universe

    @staticmethod
    def get_date_range(
        start: Optional[datetime] = None, end: Optional[datetime] = None
    ) -> Tuple[str, str]:
        if not end:
            end = datetime.today()
        if not start:
            start = end - relativedelta(years=20)
        date_range = pd.date_range(start=start, end=end, freq="M")

        default_start = date_range[0].strftime("%Y-%m-%d")
        default_end = date_range[-1].strftime("%Y-%m-%d")

        date_strings = [date.strftime("%Y-%m-%d") for date in date_range]

        selected_start, selected_end = st.select_slider(
            "Date Range",
            options=date_strings,
            value=(default_start, default_end),
            format_func=lambda x: f"{x}",
        )

        if selected_start >= selected_end:
            st.error("Error: The start date must be before the end date.")

        return selected_start, selected_end

    @staticmethod
    def get_bounds(
        label: str,
        min_value: float = 0,
        max_value: float = 100,
        step: float = 4,
        format_func: Callable[[Any], Any] = str,
        help: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        bounds = st.select_slider(
            label=label,
            options=np.arange(min_value, max_value + step, step),
            value=(min_value, max_value),
            format_func=format_func,
            help=help,
        )
        assert isinstance(bounds, tuple)
        low, high = bounds

        return (
            low if low != min_value else None,
            high if high != max_value else None,
        )

    @staticmethod
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
        stackgroup: Optional[str] = None,
    ):
        fig = go.Figure()

        for col in data:
            trace = go.Scatter(
                x=data.index,
                y=data[col].values,
                name=col,
                hovertemplate=hovertemplate,
                stackgroup=stackgroup,
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

    @staticmethod
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
        fig = go.Figure()

        for col in data:
            trace = go.Bar(
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

    @staticmethod
    def pie(
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

        fig.add_trace(
            go.Pie(labels=data.index, values=data.values, hovertemplate=hovertemplate)
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
        return fig
