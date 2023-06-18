"""ROBERT"""
from typing import List, Union, Set, Tuple
import numpy as np
import pandas as pd
from pkg.src.core import metrics
from pkg.src import data


__all__ = [
    "PriceMomentum1M",
    "PriceMomentum2M",
    "PriceMomentum3M",
    "PriceMomentum6M",
    "PriceMomentum9M",
    "PriceMomentum12M",
    "PriceMomentum18M",
    "PriceMomentum24M",
    "PriceMomentum36M",
    "PriceMomentum6M1M",
    "PriceMomentum9M1M",
    "PriceMomentum12M1M",
    "PriceMomentum18M1M",
    "PriceMomentum24M1M",
    "PriceMomentum36M1M",
    "PriceMomentum6M2M",
    "PriceMomentum9M2M",
    "PriceMomentum12M2M",
    "PriceMomentum18M2M",
    "PriceMomentum24M2M",
    "PriceMomentum36M2M",
    "PriceVolatility1M",
    "PriceVolatility3M",
]


# The Factors class contains a method that returns a DataFrame of standard
# percentiles for the factors.
class Factors(object):
    factors = pd.DataFrame()

    def __init__(self, tickers: Union[str, List, Set, Tuple]) -> None:
        self.tickers = tickers

    def __repr__(self) -> str:
        return "Base Factors"

    def __str__(self) -> str:
        return "Base Factors"

    @property
    def standard_percentile(self) -> pd.DataFrame:
        return self.factors.apply(metrics.to_standard_percentile, axis=1)


doc_string = """Price Mommentum is calculate {months}"""


class PriceMomentum(Factors):
    months = 1
    skip_months = 0
    absolute = False
    __doc__ = doc_string.format(months=months)

    def __init__(self, tickers: Union[str, List, Set, Tuple]) -> None:
        super().__init__(tickers=tickers)
        self.factors = metrics.rolling.to_momentum(
            prices=data.get_prices(tickers=self.tickers),
            months=self.months,
            skip_months=self.skip_months,
            absolute=self.absolute,
        )


class PriceMomentum1M(PriceMomentum):
    months = 1


class PriceMomentum2M(PriceMomentum):
    months = 2


class PriceMomentum3M(PriceMomentum):
    months = 3


class PriceMomentum6M(PriceMomentum):
    months = 6


class PriceMomentum9M(PriceMomentum):
    months = 9


class PriceMomentum12M(PriceMomentum):
    months = 12


class PriceMomentum18M(PriceMomentum):
    months = 18


class PriceMomentum24M(PriceMomentum):
    months = 24


class PriceMomentum36M(PriceMomentum):
    months = 36


class PriceMomentum6M1M(PriceMomentum):
    months = 6
    skip_months = 1


class PriceMomentum6M2M(PriceMomentum):
    months = 6
    skip_months = 2


class PriceMomentum9M1M(PriceMomentum):
    months = 9
    skip_months = 1


class PriceMomentum9M2M(PriceMomentum):
    months = 9
    skip_months = 2


class PriceMomentum12M1M(PriceMomentum):
    months = 12
    skip_months = 1


class PriceMomentum12M2M(PriceMomentum):
    months = 12
    skip_months = 2


class PriceMomentum18M1M(PriceMomentum):
    months = 18
    skip_months = 1


class PriceMomentum18M2M(PriceMomentum):
    months = 18
    skip_months = 2


class PriceMomentum24M1M(PriceMomentum):
    months = 24
    skip_months = 1


class PriceMomentum24M2M(PriceMomentum):
    months = 24
    skip_months = 2


class PriceMomentum36M1M(PriceMomentum):
    months = 36
    skip_months = 1


class PriceMomentum36M2M(PriceMomentum):
    months = 36
    skip_months = 2


class PriceMomentumAbs12M(PriceMomentum):
    months = 12
    absolute = True


class PriceMomentumDiffusion(Factors):
    """
    Momentum Diffusion summarizes the net price momentum at different frequencies
    and is constructed to fluctuate between +1 and -1.
    When the index is +1, it implies that upward price momentum is persistent.
    The various frequencies used are 1M, 2M, 3M, 6M, 9M, 12M. Stocks wit higher
    persistence in momentum are allocated to the top portfolio.
    """

    def __init__(self, tickers: Union[str, List, Set, Tuple]) -> None:
        super().__init__(tickers=tickers)
        self.factors = (
            pd.concat(
                objs=[
                    PriceMomentum1M(tickers=self.tickers).factors.stack(),
                    PriceMomentum2M(tickers=self.tickers).factors.stack(),
                    PriceMomentum3M(tickers=self.tickers).factors.stack(),
                    PriceMomentum6M(tickers=self.tickers).factors.stack(),
                    PriceMomentum9M(tickers=self.tickers).factors.stack(),
                    PriceMomentum12M(tickers=self.tickers).factors.stack(),
                ],
                axis=1,
            )
            .apply(np.sign)
            .sum(axis=1)
            .unstack()
            .apply(np.sign)
        )

    @property
    def standard_percentile(self) -> pd.DataFrame:
        return self.factors


class PriceVolatility(Factors):
    months = 1

    def __init__(self, tickers: Union[str, List, Set, Tuple]) -> None:
        super().__init__(tickers=tickers)
        self.factors = (
            data.get_prices(tickers=self.tickers)
            .pct_change()
            .rolling(21 * self.months)
            .std()
        )


class PriceVolatility1M(PriceVolatility):
    months = 1


class PriceVolatility3M(PriceVolatility):
    months = 3


class PriceMomentumScaledbyVolatility(Factors):
    momentum_months = 1
