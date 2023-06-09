"""ROBERT"""
import os
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
from src.core import portfolios, factors, universes, regimes
from .base import Strategy, Book, Rebalancer


class MultiStrategy(dict):
    num_strategy = 1

    def load_files(self) -> "MultiStrategy":
        directory = os.path.join(os.path.dirname(__file__), "db")
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                file_path = os.path.join(directory, filename)
                with open(file=file_path, mode="r", encoding="utf-8") as file:
                    signature = json.load(file)
                universe = universes.get(signature.pop("universe"))
                portfolio = portfolios.get(signature.pop("portfolio"))
                factor = factors.MultiFactor(*signature.pop("factors", ()))
                regime = regimes.get(signature.pop("regime"))
                constraint = signature.pop("constraint")
                commission = signature.pop("commission")
                frequency = signature.pop("frequency")
                allow_fractional_shares = signature.pop("allow_fractional_shares")
                min_periods = signature.pop("min_periods")
                inception = signature.pop("inception")
                principal = signature.pop("principal")
                book = signature.pop("book", {})
                if not book:
                    book = Book(date=pd.Timestamp(inception))
                else:
                    book = Book(**book)


                name = filename.replace(".json", "")
                self.add_strategy(
                    name=name,
                    portfolio=portfolio,
                    factor=factor,
                    regime=regime,
                    constraint=constraint,
                    universe=universe,
                    commission=commission,
                    frequency=frequency,
                    allow_fractional_shares=allow_fractional_shares,
                    min_periods=min_periods,
                    inception=inception,
                    principal=principal,
                    book=book,
                )
                self[name].save(name, override=True)
        return self

    def add_strategy(
        self,
        universe: universes.Universe,
        name: Optional[str] = None,
        portfolio: portfolios.Portfolio = portfolios.EqualWeight(),
        factor: factors.MultiFactor = factors.MultiFactor(),
        regime: regimes.Regime = regimes.OneRegime(),
        constraint: Optional[Dict] = None,
        inception: Optional[str] = None,
        frequency: str = "M",
        commission: int = 10,
        min_periods: int = 2,
        allow_fractional_shares: bool = False,
        principal: int = 10_000,
        book: Optional[Book] = None,
    ) -> "MultiStrategy":
        # check strategy name
        if name is None:
            name = f"Strategy({self.num_strategy})"
        if name in self:
            raise NameError(f"{name} already found in the mult-strategy.")

        strategy = Strategy(
            rebalancer=Rebalancer(),
            universe=universe,
            frequency=frequency,
            inception=inception,
            min_periods=min_periods,
            principal=principal,
            allow_fractional_shares=allow_fractional_shares,
            commission=commission,
            portoflio=portfolio,
            constraint=constraint,
            factor=factor,
            regime=regime,
        )
        if book is not None:
            strategy.book = book
        strategy.simulate()
        self[name] = strategy
        return self

    @property
    def performance(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: strategy.performance for name, strategy in self.items()}
        )

    @property
    def drawdowns(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: strategy.drawdown for name, strategy in self.items()}
        )

    @property
    def analytics(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: strategy.analytics for name, strategy in self.items()}
        )

    def __getitem__(self, name: str) -> Strategy:
        return super().__getitem__(name)

    def __setitem__(self, name: str, strategy: Strategy):
        # Perform custom logic here
        self.num_strategy += 1
        super().__setitem__(name, strategy)

    def items(self) -> List[Tuple[str, Strategy]]:
        return [(name, stra) for name, stra in super().items()]

    def get_signature(self, name: str) -> Dict:
        if name in self:
            return self[name].get_signature()
        raise KeyError()

    def delete(self, name: str) -> bool:
        if name not in self:
            return False
        del self[name]
        return True

    def delete_file(self, name: str) -> bool:
        file_path = Path(os.path.dirname(__file__)) / "db" / f"{name}.json"
        try:
            file_path.unlink()
            return True
        except FileNotFoundError:
            return False
        return False

    def save(self, name: str, new_name: str) -> bool:
        success = self[name].save(new_name)
        if success:
            v = self[name]
            del self[name]
            self[new_name] = v
        return success
