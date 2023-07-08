"""ROBERT"""
import re
from typing import Tuple, Optional, Callable, Any
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.core import universes
from src.backend.data import get_prices
from ..static import all_filenames


class BasePage:
    """
    This is the base page
    """

    def __init__(self) -> None:
        """
        Hello world"""
        self.load_states()
        self.load_static()
        self.load_header()
        self.load_page()
        self.load_info()

    def load_states(self) -> None:
        pass

    def load_info(self):
        st.info(self.__class__.__doc__)

    def load_page(self):
        st.warning("The Page is under construction...")

    @staticmethod
    def load_static():
        filenames = all_filenames()
        for filename in filenames:
            with open(file=filename, encoding="utf-8") as f:
                st.markdown(
                    body=f"{f.read()}"
                    if filename.endswith(".html")
                    else f"<style>{f.read()}</style>",
                    unsafe_allow_html=True,
                )

    def load_header(self):
        st.subheader(self.add_spaces_to_pascal_case(self.__class__.__name__))
        self.divider()

    @staticmethod
    def divider():
        st.markdown(
            '<hr style="margin-top: 0px; margin-bottom: 5px;">', unsafe_allow_html=True
        )

    @staticmethod
    def add_spaces_to_pascal_case(string):
        # Use regular expressions to add spaces before capital letters
        spaced_string = re.sub(r"(?<!^)(?=[A-Z])", " ", string)
        return spaced_string

    @staticmethod
    def get_universe() -> str:
        universe = str(
            st.selectbox(
                label="Universe",
                options=universes.__all__,
                index=universes.__all__.index(
                    st.session_state.get("universe", universes.__all__[0])
                ),
                help="Select investment universe.",
            )
        )
        return universe

    @staticmethod
    def get_dates(
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
            value=(
                st.session_state.get("start", default=default_start),
                st.session_state.get("end", default=default_end),
            ),
            format_func=lambda x: f"{x}",
        )
        st.session_state["start"] = selected_start
        st.session_state["end"] = selected_end

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
        xaxis_title: str = "",
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

    @staticmethod
    def get_universe_prices(universe: str) -> pd.DataFrame:
        universe_instance = getattr(universes, universe)()
        if not isinstance(universe_instance, universes.Universe):
            raise ValueError("something wrong with the universe.")
        prices = get_prices(universe_instance.get_tickers())
        return prices

    def plotly(self, fig: go.Figure, height: int =300) -> None:

        fig.update_layout(
                # plot_bgcolor='rgba(0,0,0,0)',  # Set plot background color as transparent
                # paper_bgcolor='rgba(0,0,0,0)',  # Set paper background color as transparent
                # showlegend=False,  # Hide the legend for a cleaner border look
                xaxis=dict(showgrid=False),  # Hide the x-axis gridlines
                yaxis=dict(showgrid=False),  # Hide the y-axis gridlines
                # autosize=False,  # Disable autosizing to maintain border consistency
                # width=600,  # Set the width of the chart
                height=height,  # Set the height of the chart
                margin=dict(l=20, r=20, t=20, b=20),  # Adjust the margins as needed
                # paper_bordercolor='black',  # Set the border color
                # paper_borderwidth=1  # Set the border width
            )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    def subheader(self, text: str = "") -> None:
        st.markdown(
            f"<h3 style='font-size: 1.2em;'>{text}</h3>", unsafe_allow_html=True
        )
