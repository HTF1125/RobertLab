"""ROBERT"""
from typing import Dict, Optional, Any, Tuple, Callable
import numpy as np
import pandas as pd
import streamlit as st
from src.core import strategies, portfolios, factors, universes, regimes


def get_strategy(load_files: bool = True) -> strategies.MultiStrategy:
    with st.spinner(text="Load strategies from files..."):
        if "strategy" not in st.session_state:
            multistrategy = strategies.MultiStrategy()
            if load_files:
                multistrategy.load_files()
            st.session_state["strategy"] = multistrategy
            return multistrategy
        return st.session_state["strategy"]



def get_universe() -> universes.Universe:
    universe = st.selectbox(
        label="Universe",
        options=universes.__all__,
        index=universes.__all__.index(
            st.session_state.get("universe", universes.__all__[0])
        ),
        help="Select investment universe.",
    )

    if universe is not None:
        st.session_state["universe"] = universe
        return universes.get(universe)
    raise ValueError()


def get_regime() -> regimes.Regime:
    regime = str(
        st.selectbox(
            label="Regime",
            options=regimes.__all__,
            index=regimes.__all__.index(
                st.session_state.get("regime", regimes.__all__[0])
            ),
            help="Select investment universe.",
        )
    )
    return regimes.get(regime)


class StrategyParameters:
    def __init__(
        self,
        portfolio: portfolios.Portfolio = portfolios.EqualWeight(),
        factor: factors.MultiFactor = factors.MultiFactor(),
        frequency: str = "M",
        inception: str = "2003-01-01",
        commission: int = 10,
        min_window: int = 252,
        initial_investment: int = 10_000,
        allow_fractional_shares: bool = False,
    ) -> None:
        self.portfolio = portfolio
        self.factor = factor
        self.frequency = frequency
        self.inception = inception
        self.commission = commission
        self.min_window = min_window
        self.initial_investment = initial_investment
        self.allow_fractional_shares = allow_fractional_shares
        self.fit()

    def get_portfolio(self) -> None:
        portfolio = st.selectbox(
            label="Port",
            options=portfolios.__all__,
            index=portfolios.__all__.index(self.portfolio.__class__.__name__),
            help="Select strategy's rebalancing frequency.",
        )
        if portfolio is not None:
            self.portfolio = portfolios.get(portfolio)

    def get_factor(self) -> None:
        factor = tuple(
            st.multiselect(
                label="Factor List",
                options=factors.__all__,
                default=list(self.factor.keys()),
            )
        )
        if factor:
            self.factor = factors.MultiFactor(*factor)

    def get_frequency(self) -> None:
        options = ["D", "M", "Q", "Y"]
        self.frequency = str(
            st.selectbox(
                label="Freq",
                options=options,
                index=options.index(self.frequency),
                help="Select strategy's rebalancing frequency.",
            )
        )

    def get_inception(self) -> None:
        self.inception = str(
            st.date_input(
                label="Incep",
                value=pd.Timestamp("2003-01-01"),
            )
        )

    def get_commission(self) -> None:
        self.commission = int(
            st.number_input(
                label="Comm",
                min_value=0,
                max_value=100,
                step=10,
                value=self.commission,
                help="Select strategy's trading commission in basis points.",
            )
        )

    def get_min_window(self) -> None:
        self.min_window = int(
            st.number_input(
                label="Win",
                min_value=2,
                max_value=1500,
                step=100,
                value=self.min_window,
                help="Minimum window of price data required.",
            )
        )

    def get_initial_investment(self) -> None:
        self.initial_investment = int(
            st.number_input(
                label="Invst",
                min_value=10_000,
                max_value=1_000_000,
                value=self.initial_investment,
                step=10_000,
            )
        )

    def get_allow_fractional_shares(self) -> None:
        self.allow_fractional_shares = st.checkbox(
            label="Allow fractional shares",
            value=self.allow_fractional_shares,
            help="Allow Fractional Shares Investing.",
        )

    def fit(self) -> None:
        funcs_lst = [
            [self.get_portfolio, self.get_factor],
            [
                self.get_frequency,
                self.get_min_window,
                self.get_inception,
                self.get_commission,
                self.get_initial_investment,
            ],
            [self.get_allow_fractional_shares],
        ]

        for funcs in funcs_lst:
            for col, func in zip(st.columns(len(funcs)), funcs):
                with col:
                    func()

    def dict(self) -> Dict:
        return {
            "portfolio": self.portfolio,
            "factor": self.factor,
            "frequency": self.frequency,
            "min_window": self.min_window,
            "inception": self.inception,
            "commission": self.commission,
            "initial_investment": self.initial_investment,
            "allow_fractional_shares": self.allow_fractional_shares,
        }


class StrategyConstraint:
    def __init__(
        self,
        prefix: str = "",
        universe: universes.Universe = universes.Universe(),
        leverage: int = 0,
        min_weight: Optional[int] = None,
        max_weight: Optional[int] = None,
        min_return: Optional[int] = None,
        max_return: Optional[int] = None,
        min_volatility: Optional[int] = None,
        max_volatility: Optional[int] = None,
        min_active_weight: Optional[int] = None,
        max_active_weight: Optional[int] = None,
    ) -> None:
        self.prefix = prefix
        self.leverage = leverage
        self.universe = universe
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_return = min_return
        self.max_return = max_return
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        self.min_active_weight = min_active_weight
        self.max_active_weight = max_active_weight
        self.fit()

    def get_leverage(self) -> None:
        leverage = st.select_slider(
            label="Leverage",
            options=np.arange(-100, 300 + 10, 10),
            value=self.leverage,
            key=f"{self.prefix}_leverage",
        )

        if leverage is not None:
            self.leverage = leverage

    def fit(self) -> None:
        layout = "h"
        funcs = [
            self.get_leverage,
            self.get_weight_bound,
            self.get_return_bound,
            self.get_volatility_bound,
            self.get_active_weight_bound,
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
            label="Vol.",
            min_value=0,
            max_value=30,
            step=1,
            value=(self.min_volatility or 0, self.max_volatility or 30),
            key=f"{self.prefix}_volatility",
        )

    def get_active_weight_bound(self) -> None:
        self.min_active_weight, self.max_active_weight = self.get_bounds(
            label="Act.Weight",
            min_value=0,
            max_value=200,
            step=1,
            value=(self.min_active_weight or 0, self.max_active_weight or 200),
            key=f"{self.prefix}_active_weight",
        )

    @staticmethod
    def get_bounds(
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

        if len(universe) <= 5:
            for col, asset in zip(
                st.columns(len(universe)), universe.to_dict("records")
            ):
                with col:
                    minimum, maximum = self.get_bounds(
                        label=f"{asset['ticker']}",
                        min_value=0,
                        max_value=100,
                        step=1,
                        value=(self.min_weight or 0, self.max_weight or 100),
                        key=f"{self.prefix}_{asset}",
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

    def dict(self) -> Dict:
        out = {}
        if self.leverage:
            out["leverage"] = self.leverage / 100
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
        if self.min_active_weight:
            out["min_active_weight"] = self.min_active_weight / 100
        if self.max_active_weight:
            out["max_active_weight"] = self.max_active_weight / 100
        return out
