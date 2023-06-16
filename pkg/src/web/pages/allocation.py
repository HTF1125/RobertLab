"""ROBERT"""
from typing import Dict, Tuple, List, Any, Optional, Callable
import numpy as np
import pandas as pd
import streamlit as st
from pkg.src.core import factors
from pkg.src.web import components, data, state


def get_factor_constraints():
    params = {}
    params["factors"] = st.multiselect(
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
    params["bounds"] = get_bounds(
        label="Factor Weight",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        format_func=lambda x: f"{x:.0%}",
        help="Select range for your factor weight.",
    )

    return params


def get_strategy_parameters():
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


def main():
    universe = components.get_universe()

    with st.expander(label="See universe details:"):
        st.dataframe(universe, height=150, use_container_width=True)

    with st.form("AssetAllocationForm"):
        # Backtest Parameters
        backtest_parameters = get_strategy_parameters()

        # Asset Allocation Constraints
        with st.expander(label="Asset Allocation Constraints:"):
            st.subheader("Optimizer Constraints")
            optimizer_constraints = get_optimizer_constraints()
            st.markdown("---")
            st.subheader("Specific Constraints")
            specific_constraints = get_specific_constraints(universe=universe)
            st.markdown("---")

        # Factor Implementation
        with st.expander(label="Factor Implementation:"):
            factor_constraints = get_factor_constraints()

        submitted = st.form_submit_button(label="Backtest", type="primary")
        signature = {
            "strategy": backtest_parameters,
            "constraints": {
                "optimizer": optimizer_constraints,
                "factor": factor_constraints,
                "specific": specific_constraints,
            },
        }

        st.json(signature, expanded=False)
        if submitted:
            prices = data.get_prices(tickers=universe.ticker.tolist())
            if factor_constraints["factors"]:
                factor_values = factors.multi.MultiFactors(
                    tickers=universe.ticker.tolist(),
                    factors=factor_constraints["factors"],
                ).standard_percentile
            else:
                factor_values = None

            with st.spinner(text="Backtesting in progress..."):
                state.get_backtestmanager().run(
                    **backtest_parameters,
                    optimizer_constraints=optimizer_constraints,
                    specific_constraints=specific_constraints,
                    prices=prices,
                    factor_values=factor_values,
                    factor_bounds=factor_constraints["bounds"],
                )

    st.button(
        label="Clear All Strategies",
        on_click=state.get_backtestmanager().reset_strategies,
    )
    components.performances.main()
