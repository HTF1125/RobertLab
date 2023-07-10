"""ROBERT"""
from typing import Union
import pandas as pd


class Regime:
    __states__ = ()

    def __init__(self) -> None:
        self._states = pd.Series(dtype="str")

    def get_state_by_date(self, date: Union[str, pd.Timestamp]) -> str:
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
        return str(self.get_states().loc[:date].iloc[-1])

    def get_states(self) -> pd.Series:
        start = self._states.index[0]
        idx = pd.date_range(start=start, end=pd.Timestamp("now"))
        self._states = self._states.reindex(idx).ffill().dropna()
        self._states.index.name = "Date"
        self._states.name = "State"
        return self._states
