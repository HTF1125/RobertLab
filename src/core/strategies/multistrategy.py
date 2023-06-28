"""ROBERT"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
from src import core
from src.core.portfolios import Optimizer
from src.core.universes import Universe
from src.core.benchmarks import Benchmark
from src.core.factors import MultiFactors

from .strategy import Strategy
from .rebalancer import Rebalancer
from .book import Book
from .parser import Parser


class MultiStrategy(dict):
    num_strategies = 1

    def from_files(self) -> None:
        directory = os.path.join(os.path.dirname(__file__), "db")

        # Loop through all files in the directory
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                file_path = os.path.join(directory, filename)
                with open(file=file_path, mode="r", encoding="utf-8") as file:
                    signature = json.load(file)

                book = signature.pop("book")
                if book:
                    date = pd.Timestamp(book["date"])
                    factors = signature.pop("factors")
                    if (pd.Timestamp("now") - date).days <= 5:
                        strategy = Strategy(
                            prices=pd.DataFrame(),
                            rebalance=Rebalancer(
                                optimizer=signature.pop("optimizer"),
                                factors=core.factors.MultiFactors(factors=factors),
                                optimizer_constraints=signature.pop(
                                    "optimizer_constraints"
                                ),
                                specific_constraints=signature.pop(
                                    "specific_constraints"
                                ),
                            ),
                            **signature,
                        )
                        strategy.book = Book(**book)
                        self[filename.replace(".json", "")] = strategy
                        continue

                strategy = self.prep_strategy(**signature).simulate()
                if book:
                    strategy.book = Book(**book)
                self[filename.replace(".json", "")] = strategy
                strategy.save(name=filename.replace(".json", ""), override=True)

    def prep_strategy(
        self,
        prices: Optional[pd.DataFrame] = None,
        optimizer: Union[str, Optimizer] = "EqualWeight",
        # universe & benchmark
        universe: Optional[Union[str, Universe]] = None,
        benchmark: Optional[Union[str, Benchmark]] = None,
        # rebalancer arguments
        factors: Tuple[str] = tuple(),
        optimizer_constraints: Optional[Dict[str, float]] = None,
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
        # strategy arguments
        inception: Optional[str] = None,
        frequency: str = "M",
        commission: int = 10,
        min_window: int = 2,
        allow_fractional_shares: bool = False,
        initial_investment: int = 10_000,
    ) -> Strategy:
        if prices is None:
            if universe is None:
                raise ValueError("Must provide one of prices of universe.")
            universe = Parser.get_universe(universe)
            tickers = universe.get_tickers()
            prices = universe.get_prices()
        else:
            tickers = set(prices.columns)

        strategy = Strategy(
            prices=prices,
            rebalance=Rebalancer(
                optimizer=optimizer,
                factors=MultiFactors(factors=factors).compute_standard_scaler(tickers),
                optimizer_constraints=optimizer_constraints,
                specific_constraints=specific_constraints,
            ),
            universe=universe,
            benchmark=benchmark,
            frequency=frequency,
            inception=inception,
            min_window=min_window,
            initial_investment=initial_investment,
            allow_fractional_shares=allow_fractional_shares,
            commission=commission,
        )
        return strategy

    def run(
        self,
        prices: Optional[pd.DataFrame] = None,
        optimizer: str = "EqualWeight",
        name: Optional[str] = None,
        # universe & benchmark
        universe: Optional[Union[str, Universe]] = None,
        benchmark: Optional[Union[str, Benchmark]] = None,
        # rebalancer arguments
        factors: Tuple[str] = tuple(),
        optimizer_constraints: Optional[Dict[str, float]] = None,
        specific_constraints: Optional[List[Dict[str, Any]]] = None,
        # strategy arguments
        inception: Optional[str] = None,
        frequency: str = "M",
        commission: int = 10,
        min_window: int = 2,
        allow_fractional_shares: bool = False,
        initial_investment: int = 10_000,
    ) -> Strategy:
        if name is None:
            name = f"Strategy-{self.num_strategies}"
            if name in self:
                raise ValueError("strategy `{name}` already backtested.")
        strategy = self.prep_strategy(
            prices=prices,
            optimizer=optimizer,
            universe=universe,
            benchmark=benchmark,
            factors=factors,
            optimizer_constraints=optimizer_constraints,
            specific_constraints=specific_constraints,
            inception=inception,
            frequency=frequency,
            commission=commission,
            min_window=min_window,
            allow_fractional_shares=allow_fractional_shares,
            initial_investment=initial_investment,
        )
        strategy.simulate()
        self[name] = strategy
        self.num_strategies += 1
        return strategy

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
        super().__setitem__(name, strategy)

    def items(self) -> List[Tuple[str, Strategy]]:
        return [(name, stra) for name, stra in super().items()]

    def get_signature(self, name: str) -> Dict:
        strategy = self[name]
        signature = {
            "universe": strategy.universe.__class__.__name__,
            "benchmark": strategy.benchmark.__class__.__name__,
            "min_window": strategy.min_window,
            "inception": strategy.inception,
            "frequency": strategy.frequency,
            "commission": strategy.commission,
            "allow_fractional_shares": strategy.allow_fractional_shares,
        }
        if isinstance(strategy.rebalance, Rebalancer):
            signature.update(strategy.rebalance.get_signature())
        signature["book"] = strategy.book.dict()
        return signature

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