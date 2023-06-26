"""ROBERT"""
import os
import json
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from src.backend import core
from .strategy import Strategy, ModuleParser
from .rebalancer import Rebalancer
from .book import Book


class MultiStrategy(dict):
    def from_file(self, name: str, **kwargs) -> Strategy:
        try:
            file = os.path.join(os.path.dirname(__file__), "db", f"{name}.json")
            with open(file=file, mode="r", encoding="utf-8") as file:
                signiture = json.load(file)

            book = signiture.pop("book")
            if book:
                date = pd.Timestamp(book["date"])
                if (pd.Timestamp("now") - date).days <= 5:
                    strategy = Strategy(
                        prices=pd.DataFrame(),
                        rebalance=Rebalancer(),
                        inception=signiture["inception"],
                        )
                    strategy.book = Book(**book)
                    self[name] = strategy
                    return strategy

            strategy = self.prep_strategy(**signiture).simulate()
            if book:
                strategy.book = Book(**book)
            self[name] = strategy
            return strategy

        except FileNotFoundError:
            strategy = self.run(name=name, **kwargs)

        self[name] = strategy
        self.save(name=name)
        return strategy

    def prep_strategy(
        self,
        prices: Optional[pd.DataFrame] = None,
        optimizer: str = "EqualWeight",
        name: Optional[str] = None,
        # universe & benchmark
        universe: Optional[str] = None,
        benchmark: Optional[str] = None,
        # rebalancer arguments
        factors: Optional[Tuple[str]] = None,
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
        tickers = (
            set(prices.columns)
            if isinstance(prices, pd.DataFrame)
            else ModuleParser.parse_universe(universe).get_tickers()
        )

        strategy = Strategy.from_universe(
            rebalance=Rebalancer(
                optimizer=optimizer,
                factors=core.factors.MultiFactors(tickers=tickers, factors=factors)
                if factors
                else None,
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
        universe: Optional[str] = None,
        benchmark: Optional[str] = None,
        # rebalancer arguments
        factors: Optional[Tuple[str]] = None,
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
            name = f"Strategy-{len(self) + 1}"
            if name in self:
                raise ValueError("strategy `{name}` already backtested.")
        strategy = self.prep_strategy(
            prices=prices,
            optimizer=optimizer,
            name=name,
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

    def get_siginiture(self, name: str) -> Dict:
        strategy = self[name]
        signiture = {
            "universe": strategy.universe.__class__.__name__,
            "benchmark": strategy.benchmark.__class__.__name__,
            "min_window": strategy.min_window,
            "inception": strategy.inception,
            "frequency": strategy.frequency,
            "commission": strategy.commission,
            "allow_fractional_shares": strategy.allow_fractional_shares,
        }
        if isinstance(strategy.rebalance, Rebalancer):
            signiture.update(strategy.rebalance.get_signiture())
        signiture["book"] = strategy.book.dict()
        return signiture

    def save(self, name: str) -> None:
        signiture = self.get_siginiture(name)
        file = os.path.join(os.path.dirname(__file__), "db", f"{name}.json")
        # Save the dictionary to JSON file
        try:
            with open(file=file, mode="w", encoding="utf-8") as file:
                json.dump(signiture, file)
        except:
            os.remove(file)

