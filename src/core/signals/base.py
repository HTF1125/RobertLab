from datetime import datetime
import numpy as np
import pandas as pd
from statsmodels.api import tsa
import pandas_datareader as pdr


class Signal:
    """signal"""

    __regime__ = {}

    def __init__(self) -> None:
        self.states: pd.Series = pd.Series(dtype=str)
        self.process()

    def process(self) -> None:
        """process data to signal"""
        raise NotImplementedError(
            "user must implement `process` method to make signal."
        )

    def add_states(self, idx: pd.Index) -> pd.Series:
        """assign the state column to the pandas"""
        return self.states.reindex(idx).ffill()

    def expected_returns_by_states(
        self, prices: pd.DataFrame, frequency: str = "M"
    ) -> pd.DataFrame:
        """calculate expected return by states"""
        fwd_return = (
            prices.resample(rule=frequency).last().ffill().pct_change().shift(-1)
        )
        fwd_return["states"] = self.add_states(fwd_return.index)
        grouped = fwd_return.groupby(by="states").mean()
        return grouped

    def get_state(self, date: str) -> str:
        return self.states.resample("D").last().ffill().loc[date]


def get_oecd_us_leading_indicator(meta: str = "USALOLITONOSTSAM") -> pd.DataFrame:
    data = pdr.DataReader(
        name=meta, data_source="fred", start=datetime(1900, 1, 1)
    ).astype(float)
    return data


class OECDUSLEIHP(Signal):
    """OECD US Leading Economic Indicator Signal"""

    @classmethod
    def from_fred_data(cls, **kwargs) -> "OECDUSLEIHP":
        """"""
        return cls(data=get_oecd_us_leading_indicator(), **kwargs)

    def __init__(
        self,
        data: pd.DataFrame,
        lamb: int = 0,
        min_periods: int = 12,
        months_offset: int = 1,
        resample_by: str = "M",
    ) -> None:
        self.data = data
        self.lamb = lamb
        self.min_periods = min_periods
        self.months_offset = months_offset
        self.resample_by = resample_by
        super().__init__()

    def process(self) -> None:
        """process signals"""
        # process data for processing.

        if self.resample_by:
            self.data = self.data.resample(self.resample_by).last().dropna()
        self.data.index = self.data.index + pd.DateOffset(months=self.months_offset)

        data = self.data - 100

        for idx, date in enumerate(self.data.index):
            if idx < self.min_periods:
                continue
            _, trend = tsa.filters.hpfilter(x=data.loc[:date].values, lamb=self.lamb)
            level, direction = trend[-1], np.diff(trend)[-1]
            if level >= 0 and direction >= 0:
                state = "expansion"
            elif level <= 0 and direction >= 0:
                state = "recovery"
            elif level <= 0 and direction <= 0:
                state = "contraction"
            elif level >= 0 and direction <= 0:
                state = "slowdown"
            else:
                raise ValueError("???")

            self.states.loc[date] = state
