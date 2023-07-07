from typing import Tuple, Dict, Optional, Any, Callable
import numpy as np
import pandas as pd
import streamlit as st
from src.core import factors, portfolios


class PortfolioModelSet:
    def __init__(self, prefix: str = "", **kwargs) -> None:
        self.prefix = prefix
        self.min_weight: Optional[int] = kwargs.get("min_weight", None)
        self.max_weight: Optional[int] = kwargs.get("max_weight", None)
        self.min_return: Optional[int] = kwargs.get("min_return", None)
        self.max_return: Optional[int] = kwargs.get("max_return", None)
        self.min_volatility: Optional[int] = kwargs.get("min_volatility", None)
        self.max_volatility: Optional[int] = kwargs.get("max_volatility", None)
        self.min_active_weight: Optional[int] = kwargs.get("min_active_weight", None)
        self.max_active_weight: Optional[int] = kwargs.get("max_active_weight", None)
        self.parameters = {}

    def get_factor(self) -> None:
        new_factor = tuple(
            st.multiselect(
                label="Factor List",
                options=factors.__all__,
                key=f"{self.prefix}_factor_select",
                default=list(self.factor.keys())
                if isinstance(self.factor, factors.MultiFactor)
                else None,
            )
        )

        if isinstance(self.factor, factors.MultiFactor):
            if set(self.factor.keys()) == set(new_factor):
                return
        self.factor = factors.MultiFactor(*new_factor)

        # if self.factor is None:
        #     self.factor = new_factor
        # if new_factor.keys() == self.factor.keys():
        #     del new_factor

    def get_leverage(self):
        self.leverage = float(
            st.number_input(
                label="Levg",
                min_value=0.0,
                max_value=4.0,
                value=self.leverage,
                step=0.10,
                format="%.2f",
                key=f"{self.prefix}_leverage_select",
            )
        )

    def get_portfolio(self) -> None:
        new_portfolio = portfolios.get(
            str(
                st.selectbox(
                    label="Port",
                    key=f"{self.prefix}_portfolio_select",
                    options=portfolios.__all__,
                    help="Select your portfolio model.",
                    index=portfolios.__all__.index(
                        str(self.portfolio.__class__.__name__)
                    )
                    if self.portfolio is not None
                    else 0,
                )
            )
        )
        if self.portfolio is None:
            self.portfolio = new_portfolio
        if isinstance(new_portfolio, self.portfolio.__class__):
            del new_portfolio

    def get_min_weight(self) -> None:
        self.min_weight = st.number_input(
            label="Min.W",
            min_value=-100,
            max_value=100,
            value=self.min_weight,
            key=f"{self.prefix}_min_weight",
        )

    def get_max_weight(self) -> None:
        self.max_weight = st.number_input(
            label="Max.W",
            min_value=0,
            max_value=300,
            value=self.max_weight,
            key=f"{self.prefix}_max_weight",
        )

    def get_min_return(self) -> None:
        self.min_return = st.number_input(
            label="Min.R",
            min_value=0,
            max_value=30,
            value=self.min_return,
            key=f"{self.prefix}_min_return",
        )

    def get_max_return(self) -> None:
        self.max_return = st.number_input(
            label="Max.R",
            min_value=0,
            max_value=30,
            value=self.max_return,
            key=f"{self.prefix}_max_return",
        )

    def get_min_volatility(self) -> None:
        self.min_volatility = st.number_input(
            label="Min.V",
            min_value=0,
            max_value=30,
            value=self.min_volatility,
            key=f"{self.prefix}_min_volatility",
        )

    def get_max_volatility(self) -> None:
        self.max_volatility = st.number_input(
            label="Max.V",
            min_value=0,
            max_value=30,
            value=self.max_volatility,
            key=f"{self.prefix}_max_volatility",
        )

    def get_min_active_weight(self) -> None:
        self.min_active_weight = st.number_input(
            label="Min.AW",
            min_value=0,
            max_value=100,
            value=self.min_active_weight,
            key=f"{self.prefix}_min_active_weight",
        )

    def get_max_active_weight(self) -> None:
        self.max_active_weight = st.number_input(
            label="Max.AW",
            min_value=0,
            max_value=200,
            value=self.max_active_weight,
            key=f"{self.prefix}_max_active_weight",
        )

    def get_constraint(self, layout: str = "v") -> Dict:
        funcs = [
            self.get_weight_bound,
            self.get_return_bound,
            self.get_volatility_bound,

        ]
        if layout == "h":
            funcss = [
                funcs[i : i + len(funcs)] for i in range(0, len(funcs), len(funcs))
            ]
        else:
            funcss = [funcs[i : i + 2] for i in range(0, len(funcs), 2)]

        for funcs in funcss:
            for col, func in zip(st.columns(len(funcs)), funcs):
                with col:
                    func()

        out = {}
        if self.min_weight:
            out["min_weight"] = self.min_weight / 100
        if self.max_weight:
            out["max_weight"] = self.max_weight / 100
        if self.min_return:
            out["min_return"] = self.min_return / 100
        if self.max_return:
            out["max_return"] = self.max_return / 100
        if self.min_volatility:
            out["min_volatility"] = self.min_volatility / 100
        if self.max_volatility:
            out["max_volatility"] = self.max_volatility / 100
        return out

    def get_asset_constraints(self, universe: pd.DataFrame):
        assetclasses = universe.assetclass.unique()

        constraints = []

        for col, assetclass in zip(st.columns(len(assetclasses)), assetclasses):
            with col:
                minimum, maximum = self.get_bounds(
                    label=f"{assetclass}",
                    min_value=0,
                    max_value=100,
                    step=1,
                    value=(self.min_weight or 0, self.max_weight or 100),
                    key=f"{self.prefix}_{assetclass}",
                )

            if minimum is None and maximum is None:
                continue
            constraints.append(
                {
                    "assets": list(
                        universe[universe["assetclass"] == assetclass]["ticker"]
                    ),
                    "bounds": (
                        minimum / 100 if minimum else None,
                        maximum / 100 if maximum else None,
                    ),
                }
            )

        return constraints

    def get_weight_bound(self) -> None:
        self.min_weight, self.max_weight = self.get_bounds(
            label="Weight",
            min_value=0,
            max_value=100,
            step=1,
            value=(self.min_weight or 0, self.max_weight or 100),
            key=f"{self.prefix}_weight",
        )

    def get_return_bound(self) -> None:
        self.min_return, self.max_return = self.get_bounds(
            label="Return",
            min_value=0,
            max_value=30,
            step=1,
            value=(self.min_return or 0, self.max_return or 30),
            key=f"{self.prefix}_return",
        )

    def get_volatility_bound(self) -> None:
        self.min_volatility, self.max_volatility = self.get_bounds(
            label="volatility",
            min_value=0,
            max_value=30,
            step=1,
            value=(self.min_volatility or 0, self.max_volatility or 30),
            key=f"{self.prefix}_volatility",
        )

    def get_bounds(
        self,
        label: str,
        min_value: float = 0,
        max_value: float = 100,
        step: float = 4,
        value: Tuple[int, int] = (0, 100),
        format_func: Callable[[Any], Any] = str,
        key: Optional[str] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        bounds = st.select_slider(
            label=label,
            options=np.arange(min_value, max_value + step, step),
            value=value,
            format_func=format_func,
            key=key,
        )
        low, high = bounds

        return (
            low if low != value[0] else None,
            high if high != value[1] else None,
        )
