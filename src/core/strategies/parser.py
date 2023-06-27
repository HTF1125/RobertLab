from typing import Union, Type
from src.core import universes, benchmarks, portfolios


class Parser:
    @staticmethod
    def get_universe(
        universe: Union[str, universes.Universe, Type[universes.Universe]]
    ) -> universes.Universe:
        if not isinstance(universe, str):
            if isinstance(universe, type):
                if issubclass(universe, universes.Universe):
                    return universe()
                raise TypeError(f"Invalid benchmark type: {type(universe)}")
            if issubclass(universe.__class__, universes.Universe):
                return universe
        try:
            universe_cls = getattr(universes, str(universe))
        except AttributeError as exc:
            raise ValueError(f"Invalid universe class: {universe}") from exc
        return universe_cls()

    @staticmethod
    def get_benchmark(
        benchmark: Union[str, benchmarks.Benchmark, Type[benchmarks.Benchmark]],
    ) -> benchmarks.Benchmark:
        if not isinstance(benchmark, str):
            if isinstance(benchmark, type):
                if issubclass(benchmark, benchmarks.Benchmark):
                    return benchmark()
                raise TypeError(f"Invalid benchmark type: {type(benchmark)}")
            if issubclass(benchmark.__class__, benchmarks.Benchmark):
                return benchmark
        try:
            benchmark_cls = getattr(benchmarks, str(benchmark))
        except AttributeError as exc:
            raise ValueError(f"Invalid benchmark class: {benchmark}") from exc
        return benchmark_cls()

    @staticmethod
    def get_optimizer(
        optimizer: Union[str, portfolios.Optimizer, Type[portfolios.Optimizer]],
    ) -> portfolios.Optimizer:
        if not isinstance(optimizer, str):
            if isinstance(optimizer, type):
                if issubclass(optimizer, portfolios.Optimizer):
                    return optimizer()
                raise TypeError(f"Invalid benchmark type: {type(optimizer)}")
            if issubclass(optimizer.__class__, portfolios.Optimizer):
                return optimizer
            raise TypeError(f"Invalid benchmark type: {type(optimizer)}")
        try:
            optimizer_cls = getattr(portfolios, str(optimizer))
        except AttributeError as exc:
            raise ValueError(f"Invalid benchmark class: {optimizer}") from exc
        return optimizer_cls()
