import pandas as pd
from src.backend import data
from .base import Regime


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


class UsLeadingIndicator(Regime):
    states = ("UpRoC", "DownRoC")

    def fit(self):
        self.data = data.get_macro("USALOLITONOSTSAM").resample("M").last().iloc[:, 0]
        self.data.index = self.data.index + pd.DateOffset(months=1)

    def get_state(self, date=None):
        if self.data.empty:
            self.fit()

        if date is not None:
            d = self.data.loc[:date]
        else:
            d = self.data
        rate_of_change = d.diff().diff().iloc[-1]

        if rate_of_change > 0:
            return "UpRoC"
        return "DownRoC"


