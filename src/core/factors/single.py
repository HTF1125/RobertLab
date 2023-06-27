"""ROBERT"""
from typing import List, Union, Set, Tuple
import numpy as np
import pandas as pd
from src.core import metrics
from src.backend import data


class Factor(object):
    factors = pd.DataFrame()

    def __repr__(self) -> str:
        return "Base Factors"

    def __str__(self) -> str:
        return "Base Factors"

    def standard_scaler(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        factors = self.compute(tickers)
        return factors.apply(metrics.to_standard_scaler, axis=1)

    def compute(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        raise NotImplementedError("Yout must implement `compute` method.")


class PriceMomentum(Factor):
    months = 1
    skip_months = 0
    absolute = False

    def compute(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        return metrics.rolling.to_momentum(
            prices=data.get_prices(tickers),
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


class PriceMomentumDiffusion(Factor):
    """
    Momentum Diffusion summarizes the net price momentum at different frequencies
    and is constructed to fluctuate between +1 and -1.
    When the index is +1, it implies that upward price momentum is persistent.
    The various frequencies used are 1M, 2M, 3M, 6M, 9M, 12M. Stocks wit higher
    persistence in momentum are allocated to the top portfolio.
    """

    def compute(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        return (
            pd.concat(
                objs=[
                    PriceMomentum1M().compute(tickers).stack(),
                    PriceMomentum2M().compute(tickers).stack(),
                    PriceMomentum3M().compute(tickers).stack(),
                    PriceMomentum6M().compute(tickers).stack(),
                    PriceMomentum9M().compute(tickers).stack(),
                    PriceMomentum12M().compute(tickers).stack(),
                ],
                axis=1,
            )
            .apply(np.sign)
            .sum(axis=1)
            .unstack()
            .apply(np.sign)
        )

    def standard_scaler(self, tickers) -> pd.DataFrame:
        return self.compute(tickers)


class PriceVolatility(Factor):
    months = 1

    def compute(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        prices = data.get_prices(tickers)
        pri_return = prices.pct_change()
        std = pri_return.rolling(21 * self.months).std()
        return std


class PriceVolatility1M(PriceVolatility):
    months = 1


class PriceVolatility3M(PriceVolatility):
    months = 3


# class PriceMomentumScaledbyVolatility(Factors):
#     momentum_months = 1


class VCV(Factor):
    months = 1

    def compute(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        import yfinance as yf

        volume = yf.download(tickers, progress=False)["Volume"]
        mean = volume.rolling(self.months * 21).mean()
        std = volume.rolling(self.months * 21).std()
        return -std / mean


class VolumeCoefficientOfVariation1M(VCV):
    months = 1


class VolumeCoefficientOfVariation3M(VCV):
    months = 3
