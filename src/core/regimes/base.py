from typing import Dict
import pandas as pd

class Regime:
    states = ()

    def __init__(self) -> None:
        self.data = pd.DataFrame({})
        self.constraint = {}

    def get_data(self) -> None:
        # self.data = data.get_prices("^VIX")
        pass

    def set_state_constraint(self, state: str, constraint: Dict) -> None:
        self.constraint[state] = constraint

    def get_state(self, date=None) -> str:
        return ""

    def get_portfolio_constraint(self, date=None) -> Dict:
        state = self.get_state(date)
        return self.constraint.get(state, {})
