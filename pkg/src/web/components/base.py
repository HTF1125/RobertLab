"""ROBERT"""
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import streamlit as st
from pkg.src import data
from pkg.src.core import portfolios, strategies, factors


def add_social():
    # Social Icons
    social_icons = {
        # Platform: [URL, Icon]
        "LinkedIn": [
            "https://www.linkedin.com/in/htf1125",
            "https://cdn-icons-png.flaticon.com/512/174/174857.png",
        ],
        "GitHub": [
            "https://github.com/htf1125",
            "https://icon-library.com/images/github-icon-white/github-icon-white-6.jpg",
        ],
    }

    social_icons_html = [
        f"""
        <a href='{social_icons[platform][0]}'>
            <img
                src='{social_icons[platform][1]}'
                alt='{platform}'
                height='25px'
                width='25px'
                style='margin-right: 10px; margin-left: 10px;'
            >
        </a>
        """
        for platform in social_icons
    ]

    st.write(
        f"""
        <div style="position: absolute; top: 0; left: 0;">
            {''.join(social_icons_html)}

        </div>
        """,
        unsafe_allow_html=True,
    )


def add_badges():
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
        }
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


def get_universe(show: bool = False) -> pd.DataFrame:
    universe = data.get_universe()
    selected = st.selectbox(
        label="Select Investment Universe",
        options=universe.universe.unique(),
        help="Select strategy's investment universe.",
    )
    universe = universe[universe.universe == selected]
    if show:
        with st.expander(label="See universe details:"):
            st.dataframe(universe, height=150, use_container_width=True)

    return universe


def get_date_range() -> Tuple[str, str]:
    end = datetime.today()
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


def get_start() -> str:
    return str(
        st.date_input(
            label="Start",
            value=datetime.today() - relativedelta(years=20),
            help="Start date of the strategy backtest.",
        )
    )


def get_end() -> str:
    return str(
        st.date_input(
            label="End",
            value=datetime.today(),
            help="End date of the strategy backtest.",
        )
    )


def get_frequency() -> str:
    options = ["D", "M", "Q", "Y"]
    return str(
        st.selectbox(
            label="Freq",
            options=options,
            index=options.index("M"),
            help="Select strategy's rebalancing frequency.",
        )
    )


def get_commission() -> int:
    return int(
        st.number_input(
            label="Comm",
            min_value=0,
            max_value=100,
            step=10,
            value=10,
            help="Select strategy's trading commission in basis points.",
        )
    )


def get_benchmark() -> str:
    benchmark = st.selectbox(
        label="BM",
        options=[
            strategies.benchmarks.Global64.__name__,
            strategies.benchmarks.UnitedStates64.__name__,
        ],
        help="Select strategy's benchmark.",
    )

    if benchmark is None:
        raise ValueError()
    return benchmark


def get_optimizer() -> str:
    optimizer = st.selectbox(
        label="Opt",
        options=[
            portfolios.EqualWeight.__name__,
            portfolios.MaxReturn.__name__,
            portfolios.MaxSharpe.__name__,
            portfolios.MinVolatility.__name__,
            portfolios.MinCorrelation.__name__,
            portfolios.InverseVariance.__name__,
            portfolios.RiskParity.__name__,
            portfolios.HRP.__name__,
            portfolios.HERC.__name__,
        ],
        help="Select strategy's rebalancing frequency.",
    )

    if optimizer is None:
        raise ValueError()
    return optimizer


def get_strategy_general_params():
    task = [
        get_universe,
        get_start,
        get_end,
        get_optimizer,
        get_frequency,
        get_commission,
    ]

    cache = {}
    for idx, col in enumerate(st.columns([1] * len(task))):
        with col:
            cache[str(task[idx].__name__).replace("get_", "")] = task[idx]()
    return cache



def get_percentile() -> int:
    return int(
        st.number_input(
            label="Factor Percentile", min_value=0, max_value=100, step=5, value=50
        )
    )



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


def get_factor_constraints() -> Dict:
    params = {}
    c1, c2 = st.columns([4, 1], gap="large")
    params["factors"] = c1.multiselect(
        label="Factor List",
        options=[
            factors.PriceMomentum1M.__name__,
            factors.PriceMomentum2M.__name__,
            factors.PriceMomentum3M.__name__,
            factors.PriceMomentum6M.__name__,
            factors.PriceMomentum9M.__name__,
            factors.PriceMomentum12M.__name__,
            factors.PriceMomentum24M.__name__,
            factors.PriceMomentum36M.__name__,
            factors.PriceMomentum6M1M.__name__,
            factors.PriceMomentum6M2M.__name__,
            factors.PriceMomentum9M1M.__name__,
            factors.PriceMomentum9M2M.__name__,
            factors.PriceMomentum12M1M.__name__,
            factors.PriceMomentum12M2M.__name__,
            factors.PriceMomentum24M1M.__name__,
            factors.PriceMomentum24M2M.__name__,
            factors.PriceMomentum36M1M.__name__,
            factors.PriceMomentum36M2M.__name__,
            factors.PriceVolatility1M.__name__,
            factors.PriceVolatility3M.__name__,
        ],
    )
    with c2:
        params["bounds"] = get_bounds(
            label="Factor Weight",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            format_func=lambda x: f"{x:.0%}",
            help="Select range for your factor weight.",
        )
    return params


def get_optimizer_constraints():
    constraints = {}
    kwargs = [
        {
            "name": "weight",
            "label": "Weight Bounds",
            "min_value": 0.0,
            "max_value": 1.0,
            "step": 0.05,
            "format_func": lambda x: f"{x:.0%}",
        },
        {
            "name": "return",
            "label": "Return Bounds",
            "min_value": 0.0,
            "max_value": 0.3,
            "step": 0.01,
            "format_func": lambda x: f"{x:.0%}",
        },
        {
            "name": "volatility",
            "label": "Volatility Bounds",
            "min_value": 0.0,
            "max_value": 0.3,
            "step": 0.01,
            "format_func": lambda x: f"{x:.0%}",
        },
        {
            "name": "active_weight",
            "label": "Active Weight Bounds",
            "min_value": 0.0,
            "max_value": 0.3,
            "step": 0.01,
            "format_func": lambda x: f"{x:.0%}",
        },
        {
            "name": "expost_tracking_error",
            "label": "Ex-Post Tracking Error Bounds",
            "min_value": 0.0,
            "max_value": 0.1,
            "step": 0.01,
            "format_func": lambda x: f"{x:.0%}",
        },
        {
            "name": "exante_tracking_error",
            "label": "Ex-Ante Tracking Error Bounds",
            "min_value": 0.0,
            "max_value": 0.1,
            "step": 0.01,
            "format_func": lambda x: f"{x:.0%}",
        },
    ]

    cols = st.columns([1] * 4, gap="large")

    for idx, kwarg in enumerate(kwargs[:4]):
        assert isinstance(kwarg, dict)
        name = kwarg.pop("name")
        with cols[idx]:
            minimum, maximum = get_bounds(**kwarg)
            if minimum is not None:
                constraints[f"min_{name}"] = minimum
            if maximum is not None:
                constraints[f"max_{name}"] = maximum

    cols = st.columns([1] * 2, gap="large")

    for idx, kwarg in enumerate(kwargs[4:]):
        assert isinstance(kwarg, dict)
        name = kwarg.pop("name")
        with cols[idx]:
            minimum, maximum = get_bounds(**kwarg)
            if minimum is not None:
                constraints[f"min_{name}"] = minimum
            if maximum is not None:
                constraints[f"max_{name}"] = maximum

    return constraints


def get_specific_constraints(
    universe: pd.DataFrame, num_columns: int = 5
) -> List[Dict]:
    constraints = []
    asset_classes = universe["assetclass"].unique()
    final_num_columns = min(num_columns, len(asset_classes))
    cols = st.columns([1] * final_num_columns, gap="large")
    for idx, asset_class in enumerate(asset_classes):
        with cols[idx % num_columns]:
            bounds = get_bounds(
                label=asset_class,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format_func=lambda x: f"{x:.0%}",
            )
            if bounds == (None, None):
                continue
            constraint = {
                "assets": universe[
                    universe["assetclass"] == asset_class
                ].ticker.to_list(),
                "bounds": bounds,
            }
            constraints.append(constraint)
    st.markdown("---")
    final_num_columns = min(num_columns, len(universe))
    cols = st.columns([1] * final_num_columns, gap="large")
    for idx, asset in enumerate(universe.to_dict("records")):
        ticker = asset["ticker"]
        name = asset["name"]
        with cols[idx % num_columns]:
            bounds = get_bounds(
                label=ticker,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format_func=lambda x: f"{x:.0%}",
                help=name,
            )
            if bounds == (None, None):
                continue
            constraint = {
                "assets": ticker,
                "bounds": bounds,
            }
            constraints.append(constraint)
    return constraints


def get_allow_fractional_shares():
    return st.checkbox(
        label="Fractional Shares",
        value=False,
        help="Allow Fractional Shares Investing.",
    )
