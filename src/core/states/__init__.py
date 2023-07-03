import pandas as pd
from src.backend import data


class State:
    states = ()

    def __init__(self) -> None:
        self.data = pd.DataFrame()

    def get_data(self) -> None:
        self.data = data.get_prices("^VIX")


__all__ = ["FixedOneState", "FixedTwoState"]


class FixedOneState(State):
    states = ("OneState",)


class FixedTwoState(State):
    states = ("OneState", "TwoState")
