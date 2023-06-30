"""ROBERT"""
import os
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union, Type
import pandas as pd
from src.core import portfolios, factors, universes, benchmarks
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
                # benchmark = benchmarks.get(signature.pop("benchmark"))
                optimizer = portfolios.get(signature.pop("optimizer"))
                factor = signature.pop("factors", ())
                optimizer_constraints = signature.pop("optimizer_constraints")
                specific_constraints = signature.pop("specific_constraints")
                commission = signature.pop("commission")
                frequency = signature.pop("frequency")
                allow_fractional_shares = signature.pop("allow_fractional_shares")
                min_window = signature.pop("min_window")
                inception = signature.pop("inception")
                book = Book(**signature.pop("book"))
                self.add_strategy(
                    name=filename.replace(".json", ""),
                    optimizer=optimizer,
                    factor=factor,
                    optimizer_constraints=optimizer_constraints,
                    specific_constraints=specific_constraints,
                    universe=universe,
                    # benchmark=benchmark,
                    commission=commission,
                    frequency=frequency,
                    allow_fractional_shares=allow_fractional_shares,
                    min_window=min_window,
                    inception=inception,
                    book=book,
                )

        return self

    def add_strategy(
        self,
        universe: universes.Universe,
        # benchmark: benchmarks.Benchmark,
        name: Optional[str] = None,
        optimizer: portfolios.Optimizer = portfolios.EqualWeight(),
        factor: tuple[Union[str, factors.Factor, Type[factors.Factor]], ...] = tuple(),
        optimizer_constraints: Optional[Dict[str, float]] = None,
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
        inception: Optional[str] = None,
        frequency: str = "M",
        commission: int = 10,
        min_window: int = 2,
        allow_fractional_shares: bool = False,
        initial_investment: int = 10_000,
        book: Optional[Book] = None,
    ) -> "MultiStrategy":
        # check strategy name
        if name is None:
            name = f"Strategy({self.num_strategy})"
        if name in self:
            raise NameError(f"{name} already found in the mult-strategy.")

        # benchmark.inception = inception

        strategy = Strategy(
            rebalancer=Rebalancer(),
            universe=universe,
            # benchmark=benchmark,
            frequency=frequency,
            inception=inception,
            min_window=min_window,
            initial_investment=initial_investment,
            allow_fractional_shares=allow_fractional_shares,
            commission=commission,
            optimizer=optimizer,
            optimizer_constraints=optimizer_constraints,
            specific_constraints=specific_constraints,
            factor=factors.MultiFactor(*factor),
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

    @property
    def performance_alpha(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: strategy.performance_alpha for name, strategy in self.items()}
        )

    def delete(self, name: str) -> bool:
        if name not in self:
            return False
        del self[name]
        file_path = Path(os.path.dirname(__file__)) / "db" / f"{name}.json"
        try:
            file_path.unlink()
        except FileNotFoundError:
            pass
        return True

    def save(self, name: str, new_name: str) -> bool:
        return self[name].save(new_name)
