from typing import Union
import pandas as pd
from src.backend import data
from .base import Regime


class OneRegime(Regime):
    __states__ = ("AllState",)

    def get_state_by_date(self, date: Union[str, pd.Timestamp]) -> str:
        return "AllState"


    def get_states(self) -> pd.Series:
        return pd.Series(
            data="AllState",
            index=pd.date_range(start="1990-1-1", end=pd.Timestamp("now"), freq="D"),
        )


class VolatilityRegime(Regime):
    __states__ = ("NormalVol", "ExtremeVol")

    def get_states(self) -> pd.Series:
        if self._states.empty:
            vix = data.get_price("^VIX").ewm(21).mean()
            roll = vix.rolling(252 * 5)
            mean = roll.mean()
            std = roll.std()
            score = (vix - mean) / std
            score = score.abs().dropna()
            states = score.map(
                lambda x: self.__states__[0] if x < 0.8 else self.__states__[1]
            )
            self._states = states
        return super().get_states()


class UsLeiRegime(Regime):
    __states__ = ("Recovery", "Contraction")

    def get_states(self) -> pd.Series:
        if self._states.empty:
            d = data.get_macro("USALOLITONOSTSAM").resample("M").last().iloc[:, 0]
            d.index = d.index + pd.DateOffset(months=1)
            rate_of_change = d.diff().diff()
            self._states = rate_of_change.map(
                lambda x: self.__states__[0] if x > 0 else self.__states__[1]
            )
        return super().get_states()
