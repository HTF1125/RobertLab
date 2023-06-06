"""ROBERT"""
from typing import Dict, Tuple, List, Any, Optional, Callable
import numpy as np
import pandas as pd
import streamlit as st
from .. import components, data, state
from pkg.src.core import feature, metrics


def get_features_percentile():
    params = {}

    params["names"] = st.multiselect(
        label="Factor List",
        options=[func for func in dir(feature) if callable(getattr(feature, func))],
        format_func=lambda x: "".join(word.capitalize() for word in x.split("_")),
    )
    params["bounds"] = get_bounds(
        label="Factor Weight",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        format_func=lambda x: f"{x:.0%}",
        help="Select range for your factor weight."
    )

    return params


def get_base_parameters():
    basic_calls = [
        components.get_name,
        components.get_start,
        components.get_end,
        components.get_objective,
        components.get_frequency,
        components.get_commission,
    ]
    basic_cols = st.columns([1] * len(basic_calls))

    base_parameters = {}

    for col, call in zip(basic_cols, basic_calls):
        with col:
            base_parameters[call.__name__[4:]] = call()
    return base_parameters


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


def get_general_constraints():
    constraints = {}
    kwargs = [
        {
            "name": "weights_bounds",
            "label": "Weight Bounds",
            "min_value": 0.0,
            "max_value": 1.0,
            "step": 0.05,
            "format_func": lambda x: f"{x:.0%}",
        },
        {
            "name": "return_bounds",
            "label": "Return Bounds",
            "min_value": 0.0,
            "max_value": 0.3,
            "step": 0.01,
            "format_func": lambda x: f"{x:.0%}",
        },
        {
            "name": "risk_bounds",
            "label": "Risk Bounds",
            "min_value": 0.0,
            "max_value": 0.3,
            "step": 0.01,
            "format_func": lambda x: f"{x:.0%}",
        },
        {
            "name": "active_weight_bounds",
            "label": "Active Weight Bounds",
            "min_value": 0.0,
            "max_value": 0.3,
            "step": 0.01,
            "format_func": lambda x: f"{x:.0%}",
        },
        {
            "name": "expost_tracking_error_bounds",
            "label": "Ex-Post Tracking Error Bounds",
            "min_value": 0.0,
            "max_value": 0.1,
            "step": 0.01,
            "format_func": lambda x: f"{x:.0%}",
        },
        {
            "name": "exante_tracking_error_bounds",
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


def get_asset_constraints(universe: pd.DataFrame, num_columns: int = 5) -> List[Dict]:
    constraints = []
    asset_classes = universe["assetclass"].unique()
    num_columns = min(num_columns, len(asset_classes))
    cols = st.columns([1] * num_columns)
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
                "asset": universe[
                    universe["assetclass"] == asset_class
                ].ticker.to_list(),
                "bounds": bounds,
            }
            constraints.append(constraint)
    st.markdown("---")
    num_columns = min(num_columns, len(universe))
    cols = st.columns([1] * num_columns)
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
                "asset": ticker,
                "bounds": bounds,
            }
            constraints.append(constraint)
    return constraints


def main():
    universe = components.get_universe()
    prices = data.get_prices(tickers=universe.ticker.tolist())

    with st.expander(label="See universe details:"):
        st.markdown("Universe:")
        st.dataframe(universe, height=150, use_container_width=True)
        st.markdown("Prices:")
        st.dataframe(prices, height=150, use_container_width=True)

    with st.form("AssetAllocationForm"):


        # Backtest Parameters
        backtest_parameters = get_base_parameters()


        # Asset Allocation Constraints
        with st.expander(label="Asset Allocation Constraints:"):
            general_constraints = get_general_constraints()
            st.markdown("---")
            specific_constraints = get_asset_constraints(universe=universe)
            st.markdown("---")

        # Factor Implementation
        with st.expander(label="Factor Implementation:"):
            feature_constraints = get_features_percentile()


        
        submitted = st.form_submit_button(label="Backtest", type="primary")
        signature = {
                "strategy": backtest_parameters,
                "constraints" : {
                    "feature": feature_constraints,
                    "general": general_constraints,
                    "asset" : specific_constraints
                }
            }

        st.json(signature, expanded=False)
        if submitted:
            if feature_constraints["names"]:
                feature_values = data.get_factors(
                    *feature_constraints["names"],
                    tickers=universe.ticker.tolist(),
                )
            else:
                feature_values = None

            with st.spinner(text="Backtesting in progress..."):
                state.strategy.get_backtestmanager().Base(
                    **backtest_parameters,
                    prices=prices,
                    feature_values=feature_values,
                    feature_bounds=feature_constraints["bounds"],
                )

    st.button(
        label="Clear All Strategies",
        on_click=state.strategy.get_backtestmanager().reset_strategies,
    )
    components.performances.main()
