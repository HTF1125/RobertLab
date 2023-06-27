"""ROBERT"""
from typing import Dict, Optional, Any
import pandas as pd


class Records(dict):
    def __init__(self, **kwargs) -> None:
        self["value"] = kwargs.get("value", {})
        self["cash"] = kwargs.get("cash", {})
        self["allocations"] = kwargs.get("allocations", {})
        self["weights"] = kwargs.get("weights", {})
        self["shares"] = kwargs.get("shares", {})
        self["trades"] = kwargs.get("trades", {})

    @property
    def performance(self) -> pd.Series:
        perf = pd.Series(self["value"], name="performance")
        perf.index = pd.to_datetime(perf.index)
        return perf

    @property
    def cash(self) -> pd.Series:
        return pd.Series(self["cash"], name="cash")

    @property
    def allocations(self) -> pd.DataFrame:
        return pd.DataFrame(self["allocations"]).T

    @property
    def weights(self) -> pd.DataFrame:
        return pd.DataFrame(self["weights"]).T

    @property
    def trades(self) -> pd.DataFrame:
        return pd.DataFrame(self["trades"]).T


class Book:
    @classmethod
    def new(
        cls,
        inception: pd.Timestamp,
        initial_investment: float = 10_000.0,
    ) -> "Book":
        return cls(
            date=inception,
            value=initial_investment,
            cash=initial_investment,
            shares=pd.Series(dtype=float),
            weights=pd.Series(dtype=float),
            capitals=pd.Series(dtype=float),
        )

    def __init__(
        self,
        date: pd.Timestamp,
        value: float,
        cash: float,
        shares: pd.Series,
        weights: pd.Series,
        capitals: pd.Series,
        records: Optional[Dict] = None,
    ):
        self.value = value
        self.cash = cash
        self.shares = (
            shares
            if isinstance(shares, pd.Series)
            else pd.Series(shares, dtype=float)
        )
        self.weights = (
            weights
            if isinstance(weights, pd.Series)
            else pd.Series(weights, dtype=float)
        )
        self.capitals = (
            capitals
            if isinstance(capitals, pd.Series)
            else pd.Series(capitals, dtype=float)
        )
        self.records = Records(**(records or {}))
        self.date = date

    def dict(self) -> Dict:
        return {
            "date": str(self.date),
            "value": self.value,
            "cash": self.cash,
            "shares": self.shares.to_dict(),
            "weights": self.weights.to_dict(),
            "capitals": self.capitals.to_dict(),
            "records": self.records,
        }

    @property
    def date(self) -> pd.Timestamp:
        return self._date

    @date.setter
    def date(self, date: Any) -> None:
        self._date = (
            pd.Timestamp(date)
            if not isinstance(date, pd.Timestamp)
            else date
        )
        self.records["value"][str(date)] = self.value
        self.records["cash"][str(date)] = self.cash
        self.records["shares"][str(date)] = self.shares.to_dict()
        self.records["weights"][str(date)] = self.weights.to_dict()
