"""ROBERT"""
from typing import List, Union, Set, Tuple
import numpy as np
import pandas as pd
from pkg.src.core import metrics
from pkg.src import data
from .base import Factors






doc_string = \
"""Price Mommentum is calculate {months}"""


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
