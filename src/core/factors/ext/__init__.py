"""ROBERT"""
from typing import Union, List, Set, Tuple
import numpy as np
import pandas as pd
from src.backend import data
from src.core.factors.base import Factor
from src.core import metrics


import sys
from typing import Type


def get(factor: Union[str, Factor, Type[Factor]]) -> "Factor":
    # Use getattr() to get the attribute value
    try:
        if isinstance(factor, str):
            return getattr(sys.modules[__name__], factor)()
        if isinstance(factor, type) and issubclass(factor, Factor):
            return factor()
        if issubclass(factor.__class__, Factor):
            return factor
        return getattr(sys.modules[__name__], str(factor))()
    except AttributeError as exc:
        raise ValueError(f"Invalid factor: {factor}") from exc


class PxMom(Factor):
    months = 1
    skip_months = 1
    absolute = False

    def fit(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        return metrics.rolling.to_momentum(
            prices=data.get_prices(set(tickers)),
            months=self.months,
            skip_months=self.skip_months,
            absolute=self.absolute,
        )


class PxMom1M(PxMom):
    months = 1


class PxMom2M(PxMom):
    months = 2


class PxMom3M(PxMom):
    months = 3


class PxMom6M(PxMom):
    months = 6


class PxMom9M(PxMom):
    months = 9


class PxMom12M(PxMom):
    months = 12


class PxMom18M(PxMom):
    months = 18


class PxMom24M(PxMom):
    months = 24


class PxMom36M(PxMom):
    months = 36


class PxMom6M1M(PxMom):
    months = 6
    skip_months = 1


class PxMom6M2M(PxMom):
    months = 6
    skip_months = 2


class PxMom9M1M(PxMom):
    months = 9
    skip_months = 1


class PxMom9M2M(PxMom):
    months = 9
    skip_months = 2


class PxMom12M1M(PxMom):
    months = 12
    skip_months = 1


class PriceMomentum12M2M(PxMom):
    months = 12
    skip_months = 2


class PriceMomentum18M1M(PxMom):
    months = 18
    skip_months = 1


class PriceMomentum18M2M(PxMom):
    months = 18
    skip_months = 2


class PriceMomentum24M1M(PxMom):
    months = 24
    skip_months = 1


class PriceMomentum24M2M(PxMom):
    months = 24
    skip_months = 2


class PriceMomentum36M1M(PxMom):
    months = 36
    skip_months = 1


class PriceMomentum36M2M(PxMom):
    months = 36
    skip_months = 2


class PriceMomentumAbs12M(PxMom):
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

    def fit(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        return (
            pd.concat(
                objs=[
                    PxMom1M().fit(tickers).stack(),
                    PxMom2M().fit(tickers).stack(),
                    PxMom3M().fit(tickers).stack(),
                    PxMom6M().fit(tickers).stack(),
                    PxMom9M().fit(tickers).stack(),
                    PxMom12M().fit(tickers).stack(),
                ],
                axis=1,
            )
            .apply(np.sign)
            .sum(axis=1)
            .unstack()
            .apply(np.sign)
        )

    def get_factor_by_date(
        self,
        tickers: Union[str, List, Set, Tuple],
        date: str,
        method: str = "standard_scaler",
    ) -> pd.Series:
        return super().get_factor_by_date(tickers=tickers, date=date, method="raw")


class PriceRelVol1M3M(Factor):
    months = 1
    lookback_months = 3

    def fit(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        prices = data.get_prices(set(tickers))
        pri_return = prices.pct_change()
        short_std = pri_return.rolling(21 * self.months).std()
        long_std = pri_return.rolling(21 * self.lookback_months).std()
        return (short_std - long_std).dropna(how="all", axis=0)


class PriceVolatility(Factor):
    months = 1

    def fit(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        prices = data.get_prices(set(tickers))
        pri_return = prices.pct_change()
        return pri_return.rolling(21 * self.months).std()


class PriceVolatility1M(PriceVolatility):
    months = 1


class PriceVolatility3M(PriceVolatility):
    months = 3


# class PriceMomentumScaledbyVolatility(Factors):
#     momentum_months = 1


class VCV(Factor):
    months = 1

    def fit(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        volume = data.get_volumes(tickers)
        mean = volume.rolling(self.months * 21).mean()
        std = volume.rolling(self.months * 21).std()
        return -std / mean


class VolumeCoefficientOfVariation1M(VCV):
    months = 1


class VolumeCoefficientOfVariation3M(VCV):
    months = 3


class VolumeCoefficientOfVariation6M(VCV):
    months = 6


class VolumeCoefficientOfVariation12M(VCV):
    months = 12


class MACD1(Factor):
    fast: int = 12
    slow: int = 26
    signal: int = 9

    def fit(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        prices = data.get_prices(set(tickers))
        MACD = (
            +prices.ewm(span=self.fast, min_periods=self.fast).mean()
            - prices.ewm(span=self.slow, min_periods=self.slow).mean()
        )
        signal = -MACD.ewm(span=self.signal, min_periods=self.signal).mean()
        return signal


class MACD2(MACD1):
    fast: int = 50
    slow: int = 100
    signal: int = 30


class Osc(Factor):
    fast = 32
    slow = 96

    def fit(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        prices = data.get_prices(set(tickers))
        f, g = 1 - 1 / self.fast, 1 - 1 / self.slow
        osc = (
            prices.ewm(span=2 * self.fast - 1).mean()
            - prices.ewm(span=2 * self.slow - 1).mean()
        ) / np.sqrt(1.0 / (1 - f * f) - 2.0 / (1 - f * g) + 1.0 / (1 - g * g))
        return osc
