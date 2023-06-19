"""ROBERT"""
from typing import List, Dict, Any, Type, Optional, Callable, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import streamlit as st
from pkg.src import data
from pkg.src.core import portfolios, strategies, factors


def add_badges():
    tt = """
    <div style="position: absolute; top: 0; right: 0;">
        <a href="https://github.com/htf1125/RobertLab">
            <img src="https://img.shields.io/badge/GitHub-htf1125-black?logo=github">
        </a>
        <a href="https://github.com/HTF1125/RobertLab">
            <img src="https://img.shields.io/github/license/htf1125/robertlab">
        </a>
    </div>
    """
    st.markdown(tt, unsafe_allow_html=True)



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


def get_date_range():
    left, right = st.columns([1, 1])
    with left:
        s = get_start()
    with right:
        e = get_end()
    return s, e


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


def get_benchmark() -> Type[strategies.benchmarks.Benchmark]:
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
    return getattr(strategies.benchmarks, benchmark)


def get_optimizer() -> Type[portfolios.base.BaseOptimizer]:
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
            portfolios.HierarchicalRiskParity.__name__,
            portfolios.HierarchicalEqualRiskContribution.__name__,
        ],
        help="Select strategy's rebalancing frequency.",
    )

    if optimizer is None:
        raise ValueError()
    return getattr(portfolios, optimizer)


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


def get_feature() -> List[str]:
    return st.multiselect(
        label="Factors",
        options=[
            "PriceMomentum1M",
            "PriceMomentum3M",
        ],
    )


def get_percentile() -> int:
    return int(
        st.number_input(
            label="Factor Percentile", min_value=0, max_value=100, step=5, value=50
        )
    )


def get_strategy_params() -> Dict[str, Any]:
    cache = {}
    c1, c2, c3, c4, c5 = st.columns([1] * 5)
    with c1:
        cache["start"] = get_start()
    with c2:
        cache["end"] = get_end()
    with c3:
        cache["frequency"] = get_frequency()
    with c4:
        cache["commission"] = get_commission()
    with c5:
        cache["objective"] = get_optimizer()

    c1, c2 = st.columns([5, 1])

    with c1:
        cache["feature"] = get_feature()

    with c2:
        cache["percentile"] = get_percentile()

    return cache





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


def get_factor_constraints():
    params = {}
    c1, c2 = st.columns([4, 1])
    params["factors"] = c1.multiselect(
        label="Factor List",
        options=[
            factors.PriceMomentum1M,
            factors.PriceMomentum2M,
            factors.PriceMomentum3M,
            factors.PriceMomentum6M,
            factors.PriceMomentum9M,
            factors.PriceMomentum12M,
            factors.PriceMomentum24M,
            factors.PriceMomentum36M,
            factors.PriceMomentum6M1M,
            factors.PriceMomentum6M2M,
            factors.PriceMomentum9M1M,
            factors.PriceMomentum9M2M,
            factors.PriceMomentum12M1M,
            factors.PriceMomentum12M2M,
            factors.PriceMomentum24M1M,
            factors.PriceMomentum24M2M,
            factors.PriceMomentum36M1M,
            factors.PriceMomentum36M2M,
            factors.PriceVolatility1M,
            factors.PriceVolatility3M,
        ],
        format_func=lambda x: x.__name__,
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
            "name": "port_return",
            "label": "Return Bounds",
            "min_value": 0.0,
            "max_value": 0.3,
            "step": 0.01,
            "format_func": lambda x: f"{x:.0%}",
        },
        {
            "name": "port_risk",
            "label": "Risk Bounds",
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

    cols = st.columns([1] * 4)

    for idx, kwarg in enumerate(kwargs[:4]):
        assert isinstance(kwarg, dict)
        name = kwarg.pop("name")
        with cols[idx]:
            bounds = get_bounds(**kwarg)
            if bounds == (None, None):
                continue
            constraints[name] = bounds

    cols = st.columns([1] * 2)

    for idx, kwarg in enumerate(kwargs[4:]):
        assert isinstance(kwarg, dict)
        name = kwarg.pop("name")
        with cols[idx]:
            bounds = get_bounds(**kwarg)
            if bounds == (None, None):
                continue
            constraints[name] = bounds

    return constraints


def get_specific_constraints(
    universe: pd.DataFrame, num_columns: int = 5
) -> List[Dict]:
    constraints = []
    asset_classes = universe["assetclass"].unique()
    final_num_columns = min(num_columns, len(asset_classes))
    cols = st.columns([1] * final_num_columns)
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
    cols = st.columns([1] * final_num_columns)
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


from typing import Literal

import streamlit as st
from streamlit.components.v1 import html


Font = Literal[
    "Cookie",
    "Lato",
    "Arial",
    "Comic",
    "Inter",
    "Bree",
    "Poppins",
]


# # LinkedIn link
# linkedin_url = "https://www.linkedin.com/in/your-linkedin-profile"
# linkedin_icon = "https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Logo.svg.original.svg"

# # Render LinkedIn link with LinkedIn icon
# st.markdown(f'<a href="{linkedin_url}"><img src="{linkedin_icon}" width="90"></a>', unsafe_allow_html=True)



def button(
    username: str,
    floating: bool = True,
    text: str = "Test",
    emoji: str = "",
    bg_color: str = "#FFDD00",
    font: Font = "Cookie",
    font_color: str = "#000000",
    coffee_color: str = "#000000",
    width: int = 220,
):
    button = f"""
        <script type="text/javascript"
            src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js"
            # data-name="bmc-button"
            data-slug="{username}"
            data-color="{bg_color}"
            data-emoji="{emoji}"
            data-font="{font}"
            data-text="{text}"
            data-outline-color="#000000"
            data-font-color="{font_color}"
            data-coffee-color="{coffee_color}" >
        </script>
    """

    html(button, height=70, width=width)

    if floating:
        st.markdown(
            f"""
            <style>
                iframe[width="{width}"] {{
                    position: fixed;
                    bottom: 60px;
                    right: 40px;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )
