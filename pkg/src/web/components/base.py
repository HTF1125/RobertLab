"""ROBERT"""
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from pkg.src import data
from pkg.src.web import state
from pkg.src.core import portfolios


def get_universe() -> pd.DataFrame:
    universe = data.get_universe()

    selected = st.selectbox(
        label="Select Investment Universe",
        options=universe.universe.unique(),
        help="Select strategy's investment universe.",
    )
    return universe[universe.universe == selected]


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
            value=datetime.today() - timedelta(days=3650),
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


def get_objective() -> portfolios.base.BaseOptimizer:
    return st.selectbox(
        label="Obj",
        options=[
            portfolios.EqualWeight,
            portfolios.MaxReturn,
            portfolios.MaxSharpe,
            portfolios.MinVolatility,
            portfolios.MinCorrelation,
            portfolios.InverseVariance,
            portfolios.RiskParity,
            portfolios.HierarchicalRiskParity,
            portfolios.HierarchicalEqualRiskContribution,
        ],
        index=0,
        format_func=lambda x: x.__name__,
        help="Select strategy's rebalancing frequency.",
    )


def get_strategy_general_params():
    task = [
        get_universe,
        get_start,
        get_end,
        get_objective,
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


def get_name() -> str:
    return str(st.text_input(label="Strategy Name", placeholder="Example: Strategy1"))


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
        cache["objective"] = get_objective()

    c1, c2 = st.columns([5, 1])

    with c1:
        cache["feature"] = get_feature()

    with c2:
        cache["percentile"] = get_percentile()

    return cache


def get_name() -> str:
    return str(
        st.text_input(
            label="Name",
            value=f"Strategy-{state.get_backtestmanager().num_strategies}",
        )
    )
