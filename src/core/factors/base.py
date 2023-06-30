"""ROBERT"""
import sys
from typing import List, Union, Set, Tuple, Iterable
import numpy as np
import pandas as pd
from src.core import metrics
from src.backend import data


def get(factor: str) -> "Factor":
    # Use getattr() to get the attribute value
    try:
        return getattr(sys.modules[__name__], factor)()
    except AttributeError as exc:
        raise ValueError(f"Invalid factor: {factor}") from exc


class Factor(object):
    def __init__(self) -> None:
        self.factor = pd.DataFrame()

    def __repr__(self) -> str:
        return "Base Factors"

    def __str__(self) -> str:
        return "Base Factors"

    def get_factor(
        self, tickers: Union[str, List, Set, Tuple], method: str = "standard_scaler"
    ) -> pd.DataFrame:
        assert method == "standard_scaler"
        return self.standard_scaler(tickers)

    def standard_scaler(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        if self.factor.empty:
            self.compute(tickers)
        return self.factor.apply(metrics.to_standard_scaler, axis=1)

    def compute(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        raise NotImplementedError("Yout must implement `compute` method.")

    def get_factor_by_date(
        self,
        tickers: Union[str, List, Set, Tuple],
        date: str,
        method: str = "standard_scaler",
    ) -> pd.Series:
        tickers = (
            tickers
            if isinstance(tickers, (list, set, tuple))
            else tickers.replace(",", " ").split()
        )
        new_tickers = [
            ticker for ticker in tickers if ticker not in self.factor.columns
        ]
        if new_tickers:
            self.factor = pd.concat(
                objs=[self.factor, self.compute(new_tickers)], axis=1
            )
        out_factor = self.factor.ffill().loc[:date].iloc[-1]
        if method == "standard_scaler":
            return metrics.to_standard_scaler(out_factor)
        return out_factor


class PriceMomentum(Factor):
    months = 1
    skip_months = 0
    absolute = False

    def compute(self, tickers: Iterable) -> pd.DataFrame:
        return metrics.rolling.to_momentum(
            prices=data.get_prices(set(tickers)),
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

    def compute(self, tickers: Iterable) -> pd.DataFrame:
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

    def compute(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        prices = data.get_prices(set(tickers))
        pri_return = prices.pct_change()
        short_std = pri_return.rolling(21 * self.months).std()
        long_std = pri_return.rolling(21 * self.lookback_months).std()
        return (short_std - long_std).dropna(how="all", axis=0)


class PriceVolatility(Factor):
    months = 1

    def compute(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
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

    def compute(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        import yfinance as yf

        volume = yf.download(tickers, progress=False)["Volume"]
        if isinstance(volume, pd.Series):
            volume.name = tickers if isinstance(tickers, str) else str(tickers[0])
            volume = volume.to_frame()
        mean = volume.rolling(self.months * 21).mean()
        std = volume.rolling(self.months * 21).std()
        return -std / mean


class VolumeCoefficientOfVariation1M(VCV):
    months = 1


class VolumeCoefficientOfVariation3M(VCV):
    months = 3


class MACD1(Factor):
    fast: int = 12
    slow: int = 26
    signal: int = 9

    def compute(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        prices = data.get_prices(set(tickers))
        MACD = (
            +prices.ewm(span=self.fast, min_periods=self.fast).mean()
            - prices.ewm(span=self.slow, min_periods=self.slow).mean()
        )
        signal = - MACD.ewm(span=self.signal, min_periods=self.signal).mean()
        return signal


class MACD2(MACD1):
    fast: int = 50
    slow: int = 100
    signal: int = 30



class Osc(Factor):
    fast = 32
    slow = 96

    def compute(self, tickers: Union[str, List, Set, Tuple]) -> pd.DataFrame:
        prices = data.get_prices(set(tickers))
        f, g = 1 - 1 / self.fast, 1 - 1 / self.slow
        f_ewm = prices.ewm(span=2*self.fast -1).mean()
        osc = (prices.ewm(span=2*self.fast-1).mean() - prices.ewm(span=2*self.slow-1).mean())/np.sqrt(1.0 / (1 - f * f) - 2.0 / (1 - f * g) + 1.0 / (1 - g * g))
        return osc