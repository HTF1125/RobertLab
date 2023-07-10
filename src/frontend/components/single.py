"""ROBERT"""
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import streamlit as st
from src.core import universes, regimes, portfolios, factors


def get_universe(prefix: str = "") -> universes.Universe:
    universe = str(
        st.selectbox(
            label="Universe",
            options=universes.__all__,
            help="Select investment universe.",
            key=f"{prefix}_universe",
        )
    )
    return universes.get(universe)


def get_regime(prefix: str = "") -> regimes.Regime:
    regime = str(
        st.selectbox(
            label="Regime",
            options=regimes.__all__,
            help="Select investment universe.",
            key=f"{prefix}_regime",
        )
    )
    return regimes.get(regime)


def get_portfolio(prefix: str = "") -> portfolios.Portfolio:
    portfolio = str(
        st.selectbox(
            label="Portfolio",
            options=portfolios.__all__,
            help="Select construction model for portfolio.",
            key=f"{prefix}_portfolio",
        )
    )
    return portfolios.get(portfolio)


def get_factor(prefix: str = "") -> factors.MultiFactor:
    factor = tuple(
        st.multiselect(label="Factor", options=factors.__all__, key=f"{prefix}_factor")
    )
    return factors.MultiFactor(*factor)


def get_frequency(prefix: str = "") -> str:
    options = ["D", "M", "Q", "Y"]
    return str(
        st.selectbox(
            label="Frequency",
            options=options,
            help="Select strategy's rebalancing frequency.",
            index=1,
            key=f"{prefix}_frequency",
        )
    )


def get_inception(prefix: str = "") -> str:
    return str(
        st.date_input(
            label="Inception",
            value=pd.Timestamp("2003-01-01"),
            key=f"{prefix}_inception",
        )
    )


def get_commission(
    value: int = 10,
    min_value: int = 0,
    max_value: int = 100,
    step: int = 10,
    prefix: str = "",
) -> int:
    return int(
        st.number_input(
            label="Commission",
            min_value=min_value,
            max_value=max_value,
            step=step,
            value=value,
            help="Select strategy's trading commission in basis points.",
            key=f"{prefix}_commission",
        )
    )


def get_min_periods(
    value: int = 252,
    min_value: int = 2,
    max_value: int = 1500,
    step: int = 21,
    prefix: str = "",
) -> int:
    return int(
        st.number_input(
            label="Min Periods",
            min_value=min_value,
            max_value=max_value,
            step=step,
            value=value,
            help="Minimum periods of price hisotry required",
            key=f"{prefix}_min_periods",
        )
    )


def get_periods(
    value: int = 21,
    min_value: int = 21,
    max_value: int = 252,
    step: int = 21,
    prefix: str = "",
) -> int:
    return int(
        st.number_input(
            label="Periods",
            min_value=min_value,
            max_value=max_value,
            step=step,
            value=value,
            help="Forward periods for retrun",
            key=f"{prefix}_periods",
        )
    )


def get_principal(
    min_value: int = 10_000,
    max_value: int = 1_000_000,
    value: int = 10_000,
    step: int = 10_000,
) -> int:
    return int(
        st.number_input(
            label="Principal",
            min_value=min_value,
            max_value=max_value,
            value=value,
            step=step,
        )
    )


def get_allow_fractional_shares(value: bool = False, prefix: str = "") -> bool:
    return bool(
        st.checkbox(
            label="Allow fractional shares",
            value=value,
            key=f"{prefix}_allow_fractional_shares",
        )
    )


def get_leverage(
    value: int = 0,
    min_value: int = -100,
    max_value: int = 300,
    step: int = 10,
    prefix: str = "",
) -> Dict[str, float]:
    leverage = int(
        st.select_slider(
            label="Leverage",
            options=np.arange(min_value, max_value + step, step),
            value=value,
            key=f"{prefix}_leverage",
        )
    )
    out = {}
    if leverage is not None and leverage != value:
        out["leverage"] = leverage / 100.0
    return out


def get_weight_bound(
    min_value: int = 0,
    max_value: int = 100,
    step: int = 1,
    value: Tuple[int, int] = (0, 100),
    prefix: str = "",
) -> Dict[str, float]:
    low, high = st.select_slider(
        label="Weight",
        options=np.arange(min_value, max_value + step, step),
        value=value,
        key=f"{prefix}_weight_bounds",
    )

    out = {}
    if low is not None and low != min_value:
        out["min_weight"] = low / 100.0
    if high is not None and high != max_value:
        out["max_weight"] = high / 100.0
    return out


def get_return_bound(
    min_value: int = 0,
    max_value: int = 30,
    step: int = 1,
    value: Tuple[int, int] = (0, 30),
    prefix: str = "",
) -> Dict[str, float]:
    low, high = st.select_slider(
        label="Return",
        options=np.arange(min_value, max_value + step, step),
        value=value,
        key=f"{prefix}_return_bounds",
    )

    out = {}
    if low is not None and low != min_value:
        out["min_return"] = low / 100.0
    if high is not None and high != max_value:
        out["max_return"] = high / 100.0
    return out


def get_volatility_bound(
    min_value: int = 0,
    max_value: int = 30,
    step: int = 1,
    value: Tuple[int, int] = (0, 30),
    prefix: str = "",
) -> Dict[str, float]:
    low, high = st.select_slider(
        label="Vol.",
        options=np.arange(min_value, max_value + step, step),
        value=value,
        key=f"{prefix}_volatility_bounds",
    )

    out = {}
    if low is not None and low != min_value:
        out["min_volatility"] = low / 100.0
    if high is not None and high != max_value:
        out["max_volatility"] = high / 100.0

    return out


def get_portfolio_constraints(prefix: str = "") -> Dict[str, float]:
    funcs = [
        get_leverage,
        get_weight_bound,
        get_return_bound,
        get_volatility_bound,
    ]
    out = {}
    for col, func in zip(st.columns(len(funcs)), funcs):
        with col:
            out.update(func(prefix=prefix))
    return out


def get_asset_constraints(universe: universes.Universe, prefix: str = ""):
    universe_df = pd.DataFrame(universe.ASSETS)
    assetclasses = universe_df.assetclass.unique()

    constraints = []

    for col, assetclass in zip(st.columns(len(assetclasses)), assetclasses):
        with col:
            minimum, maximum = get_bounds(
                label=f"{assetclass}",
                min_value=0,
                max_value=100,
                step=1,
                value=(0, 100),
                key=f"{prefix}_{assetclass}",
            )

        if minimum is None and maximum is None:
            continue
        constraints.append(
            {
                "assets": list(
                    universe_df[universe_df["assetclass"] == assetclass]["ticker"]
                ),
                "bounds": (
                    minimum / 100 if minimum else None,
                    maximum / 100 if maximum else None,
                ),
            }
        )

    if len(universe_df) <= 5:
        for col, asset in zip(
            st.columns(len(universe_df)), universe_df.to_dict("records")
        ):
            with col:
                minimum, maximum = get_bounds(
                    label=f"{asset['ticker']}",
                    min_value=0,
                    max_value=100,
                    step=1,
                    value=(0, 100),
                    key=f"{prefix}_{asset}",
                )

            if minimum is None and maximum is None:
                continue
            constraints.append(
                {
                    "assets": list(asset),
                    "bounds": (
                        minimum / 100 if minimum else None,
                        maximum / 100 if maximum else None,
                    ),
                }
            )

    return constraints


def get_bounds(
    label: str,
    min_value: int = 0,
    max_value: int = 100,
    step: int = 4,
    value: Tuple[int, int] = (0, 100),
    key: Optional[str] = None,
) -> Tuple[Optional[int], Optional[int]]:
    bounds = st.select_slider(
        label=label,
        options=np.arange(min_value, max_value + step, step),
        value=value,
        key=key,
    )
    low, high = bounds

    return (
        low if low != value[0] else None,
        high if high != value[1] else None,
    )


def get_strategy_parameters() -> Dict:
    out = {}
    funcs_lst = [
        [get_portfolio, get_factor],
        [
            get_frequency,
            get_min_periods,
            get_inception,
            get_commission,
            get_principal,
        ],
        [get_allow_fractional_shares],
    ]

    for funcs in funcs_lst:
        for col, func in zip(st.columns(len(funcs)), funcs):
            with col:
                out.update({func.__name__.replace("get_", ""): func()})
    return out
