"""ROBERT"""
import os
import re
from typing import Tuple, Optional, Callable, Any
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import streamlit as st
from pkg.src import data


class BasePage:
    def __init__(self) -> None:
        self.load_badges()
        self.load_header()
        self.render()

    def render(self):
        st.warning("The Page is under construction...")

    @staticmethod
    def load_badges():
        # Social Icons
        social_icons = {
            # Platform: {href, src, height, width}
            "LinkedIn": {
                "href": "https://www.linkedin.com/in/htf1125",
                "src": "https://cdn-icons-png.flaticon.com/512/174/174857.png",
                "height": "25px",
                "width": "25px",
            },
            "GitHub": {
                "href": "https://github.com/htf1125",
                "src": "https://icon-library.com/images/github-icon-white/github-icon-white-6.jpg",
                "height": "25px",
                "width": "25px",
            },
            "GitLicense": {
                "href": "https://github.com/HTF1125/RobertLab",
                "src": "https://img.shields.io/github/license/htf1125/robertlab",
                "height": "25px",
                "width": "125px",
            },
        }

        social_icons_html = [
            f"""
            <a href='{platform_params["href"]}'>
                <img
                    src='{platform_params["src"]}'
                    alt='{platform}'
                    height='{platform_params["height"]}'
                    width='{platform_params["width"]}'
                    style='margin-right: 10px; margin-left: 10px; vertical-align: middle;'
                >
            </a>
            """
            for platform, platform_params in social_icons.items()
        ]

        st.write(
            f"""
            <div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%);">
                {''.join(social_icons_html)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    def load_header(self):
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