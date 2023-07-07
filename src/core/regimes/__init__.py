from typing import Dict, Optional
import pandas as pd
from src.backend import data


class Regime:
    states = ()

    def __init__(self) -> None:
        self.data = pd.DataFrame({})
        self.constraint = {}

    def get_data(self) -> None:
        self.data = data.get_prices("^VIX")

    def set_state_constraint(self, state: str, constraint: Dict) -> None:
        self.constraint[state] = constraint

    def get_state(self, date=None) -> str:
        return ""

    def get_portfolio_constraint(self, date=None) -> Dict:
        state = self.get_state(date)
        return self.constraint.get(state, {})


__all__ = ["OneRegime", "VolatilityState", "UsLeadingIndicator"]


class OneRegime(Regime):
    states = ("AllState",)

    def get_state(self, date=None) -> str:
        return "AllState"


class VolatilityState(Regime):
    states = ("NormalVol", "ExtremeVol")

    def fit(self):
        self.data = data.get_prices("^VIX")

    def get_state(self, date=None):
        if self.data.empty:
            self.fit()

        if date is not None:
            d = self.data.loc[:date]
        else:
            d = self.data
        d = d.iloc[-252 * 10 :]
        score = (d.iloc[-1] - d.mean()) / d.std()
        score = score.iloc[0]
        if score >= 0.8 or score <= -0.8:
            return "ExtremeVol"
        return "NormalVol"


import sys
from typing import Type, Union


def get(regime: Union[str, Regime, Type[Regime]]) -> Regime:
    # Use getattr() to get the attribute value
    try:
        if isinstance(regime, str):
            return getattr(sys.modules[__name__], regime)()
        if isinstance(regime, type) and issubclass(regime, Regime):
            return regime()
        if issubclass(regime.__class__, Regime):
            return regime
        return getattr(sys.modules[__name__], str(regime))()
    except AttributeError as exc:
        raise ValueError(f"Invalid factor: {regime}") from exc


class UsLeadingIndicator(Regime):
    states = ("UpRoC", "DownRoC")

    def fit(self):
        self.data = data.get_macro("USALOLITONOSTSAM")
        self.data.index = pd.to_datetime(self.data.index) + pd.DateOffset(months=1)

    def get_state(self, date=None):
        if self.data.empty:
            self.fit()

        if date is not None:
            d = self.data.loc[:date]
        else:
            d = self.data
        if d.diff().diff().iloc[-1, 0] > 0:
            return "UpRoC"
        return "DownRoC"
