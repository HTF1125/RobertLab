"""ROBERT"""
import numpy as np
import pandas as pd
from statsmodels.api import tsa
from src.backend import data


class BaseState:

    states = {
        "extreme" : lambda x: x >= 0.8 or x <= -0.8,
        "normal" : lambda x: -0.8 < x < 0.8
    }

    def __init__(self) -> None:
        self.data = pd.DataFrame()


    def fit(self) -> None:
        self.data = data.get_prices("^VIX")

    def get_state(self, idx = None) -> str:

        if self.data.empty:
            self.fit()

        if idx is not None:
            sliced = self.data.loc[:idx]
        else:
            sliced = self.data

        sliced = sliced.iloc[-252 * 10:]

        # for state, condition in self.states.items():






class Signal:
    """signal"""

    __regime__ = {}

    def __repr__(self):
        return "Signal"

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


class OECDUSLEIHP(Signal):
    """OECD US Leading Economic Indicator Signal"""

    @classmethod
    def from_fred_data(cls, **kwargs) -> Signal:
        """get the data from fred"""
        try:
            from .. import data
        except ImportError as exc:
            raise ImportError() from exc
        return cls(data=data.get_oecd_us_leading_indicator(), **kwargs)

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

    def __str__(self):
        return "OECDUSLEIHP"

    def __repr__(self):
        return "OECDUSLEIHP"

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


class VIX(Signal):
    """
    VIX is a measure of expected volatilityof the S&P 500 Index over the next 30 days.
    It is a weightedaverage of implied volatilities of SPX options that is calculated using the
    bid/ask price quotationsof the appropriate options. It is widely followed by market
    participants and has become the de-facto measure of market sentiment (investor fear/complacency)
    though over shortperiods VIX can move for technical reasons. While the level of VIX matters,
    it is the direction of VIX (rising or falling) that appears to be more predictive for the
    direction of asset valuationsincluding that of styles.A priori one should expect “risky”
    styles (what is riskyis not always easy to define) to underperform when VIX is
    rising and outperform when VIX is falling.
    """

    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:
        self.data = data
        super().__init__()

    @classmethod
    def from_fred_data(cls, **kwargs) -> Signal:
        """get the data from fred"""
        try:
            from core import data
        except ImportError as exc:
            raise ImportError() from exc
        return cls(data=data.get_vix_index(), **kwargs)

    def process(self) -> None:
        pass




class VolState:
    def __init__(self) -> None:
        self.data = pd.DataFrame()

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
            return "extreme"
        return "med"

    def get_portfolio_constraints_by_date(self, date=None):
        state = self.get_state(date)

        if state == "extreme":
            return {
                "sum_weight": 1.5,
            }
        return {
            "sum_weight": 1.0,
        }
