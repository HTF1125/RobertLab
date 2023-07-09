from typing import Union
import pandas as pd
from src.backend import data
from .base import Regime


class OneRegime(Regime):
    __states__ = ("AllState",)

    def get_state_by_date(self, date: Union[str, pd.Timestamp]) -> str:
        return "AllState"




class VolatilityRegime(Regime):
    __states__ = ("NormalVol", "ExtremeVol")

    @property
    def states(self) -> pd.Series:
        if self._states.empty:
            d = data.get_price("^VIX")

            roll = d.rolling(252 * 10)
            score = (d - roll.mean()) / roll.std()
            score = score.abs()

            self._states = (
                score.map(
                    lambda x: self.__states__[0] if x < 0.6 else self.__states__[1]
                )
                .resample("D")
                .last()
                .ffill()
                .dropna()
            )
            self._states.index.name = "Date"
            self._states.name = "State"
        return self._states


class UsLeiRegime(Regime):
    __states__ = ("Recovery", "Contraction")

    @property
    def states(self) -> pd.Series:
        if self._states.empty:
            d = data.get_macro("USALOLITONOSTSAM").resample("M").last().iloc[:, 0]
            d.index = d.index + pd.DateOffset(months=1)
            rate_of_change = d.diff().diff()
            self._states = (
                rate_of_change.map(
                    lambda x: self.__states__[0] if x > 0 else self.__states__[1]
                )
                .resample("D")
                .last()
                .ffill()
                .dropna()
            )
            self._states.index.name = "Date"
            self._states.name = "State"
        return self._states
