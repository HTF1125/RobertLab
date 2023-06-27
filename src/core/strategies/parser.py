from typing import Union, Type
from src.core import universes, benchmarks, portfolios, factors
from src.core.factors import Factor
from src.core.benchmarks import Benchmark
from src.core.universes import Universe
from src.core.portfolios import PortfolioOptimizer


class Parser:
    @staticmethod
    def get_universe(
        universe: Union[str, Universe, Type[Universe]]
    ) -> universes.Universe:
        if not isinstance(universe, str):
            if isinstance(universe, type):
                if issubclass(universe, Universe):
                    return universe()
                raise TypeError(f"Invalid benchmark type: {type(universe)}")
            if issubclass(universe.__class__, Universe):
                return universe
        try:
            universe_cls = getattr(universes, str(universe))
        except AttributeError as exc:
            raise ValueError(f"Invalid benchmark class: {universe}") from exc
        return universe_cls()

    @staticmethod
    def get_benchmark(
        benchmark: Union[str, Benchmark, Type[Benchmark]],
    ) -> Benchmark:
        if not isinstance(benchmark, str):
            if isinstance(benchmark, type):
                if issubclass(benchmark, universes.Universe):
                    return benchmark()
                raise TypeError(f"Invalid benchmark type: {type(benchmark)}")
            if issubclass(benchmark.__class__, universes.Universe):
                return benchmark
        try:
            benchmark_cls = getattr(benchmarks, str(benchmark))
        except AttributeError as exc:
            raise ValueError(f"Invalid benchmark class: {benchmark}") from exc
        return benchmark_cls()

    @staticmethod
    def get_optimizer(
        optimizer: Union[str, PortfolioOptimizer, Type[PortfolioOptimizer]],
    ) -> PortfolioOptimizer:
        if not isinstance(optimizer, str):
            if isinstance(optimizer, type):
                if issubclass(optimizer, PortfolioOptimizer):
                    return optimizer()
                raise TypeError(f"Invalid benchmark type: {type(optimizer)}")
            if issubclass(optimizer.__class__, PortfolioOptimizer):
                return optimizer
            raise TypeError(f"Invalid benchmark type: {type(optimizer)}")
        try:
            optimizer_cls = getattr(portfolios, str(optimizer))
        except AttributeError as exc:
            raise ValueError(f"Invalid benchmark class: {optimizer}") from exc
        return optimizer_cls()

    @staticmethod
    def get_factor(
        factor: Union[str, Factor, Type[Factor]],
    ) -> PortfolioOptimizer:
        if not isinstance(factor, str):
            if isinstance(factor, type):
                if issubclass(factor, Factor):
                    return factor()
                raise TypeError(f"Invalid benchmark type: {type(factor)}")
            if issubclass(factor.__class__, PortfolioOptimizer):
                return factor
            raise TypeError(f"Invalid benchmark type: {type(factor)}")
        try:
            optimizer_cls = getattr(factors, str(factor))
        except AttributeError as exc:
            raise ValueError(f"Invalid factor class: {factor}") from exc
        return optimizer_cls()
