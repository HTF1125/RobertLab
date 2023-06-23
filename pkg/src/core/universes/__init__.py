"""ROBERT"""
import os
import json
import pandas as pd
from pkg.src import data


__all__ = ["GlobalAssetAllocation", "UnitedStatesSectors"]


class Universe:
    @classmethod
    def instance(cls) -> "Universe":
        return cls()

    def __init__(self) -> None:
        file = os.path.join(os.path.dirname(__file__), "universe.json")
        with open(file=file, mode="r", encoding="utf-8") as json_file:
            self.data = pd.DataFrame(json.load(json_file)[self.__class__.__name__])

    @property
    def tickers(self) -> str:
        return ", ".join(list(self.data["ticker"]))

    @property
    def prices(self) -> pd.DataFrame:
        return data.get_prices(tickers=self.tickers)


class GlobalAssetAllocation(Universe):
    pass


class UnitedStatesSectors(Universe):
    pass
