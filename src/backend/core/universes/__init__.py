"""ROBERT"""

from typing import Tuple
import pandas as pd
from src.backend import data


__all__ = [
    "GlobalAllocation",
    "GlobalAssetAllocation",
    "UsAllocation",
    "UnitedStatesSectors",
]

class Universe:
    ASSETS = ()

    @classmethod
    def get_tickers(cls) -> Tuple:
        return tuple(pd.DataFrame(cls.ASSETS).ticker)

    @classmethod
    def get_prices(cls) -> pd.DataFrame:
        return data.get_prices(tickers=cls.get_tickers())




class GlobalAssetAllocation(Universe):
    ASSETS = (
        {
            "ticker": "SPY",
            "assetclass": "Equity",
            "name": "U.S. Stocks (S&P500)"
        },
        {
            "ticker": "EZU",
            "assetclass": "Equity",
            "name": "Europe Stocks",
        },
        {
            "ticker": "VPL",
            "assetclass": "Equity",
            "name": "Asia Pacific Stocks",
        },
        {
            "ticker": "EEM",
            "assetclass": "Equity",
            "name": "Emerging Market Stocks",
        },
        {
            "ticker": "RWR",
            "assetclass": "RealEstate",
            "name": "Global REITs"
        },
        {
            "ticker": "GLD",
            "assetclass": "Alternative",
            "name": "Gold"
        },
        {
            "ticker": "GSG",
            "assetclass": "Alternative",
            "name": "Commodities"
        },
        {
            "ticker": "IEF",
            "assetclass": "FixedIncome",
            "name": "Intermediate Treasuries (7-10Y)",
        },
        {
            "ticker": "TLT",
            "assetclass": "FixedIncome",
            "name": "Long-Term Treasuries (20+Y)",
        },
        {
            "ticker": "EMB",
            "assetclass": "FixedIncome",
            "name": "Emerging Bond (USD)"
        },
        {
            "ticker": "IGOV",
            "assetclass": "FixedIncome",
            "name": "International Treasury",
        },
        {
            "ticker": "TIP",
            "assetclass": "FixedIncome",
            "name": "Long-Term Inflation Hedged",
        },
    )


class UnitedStatesSectors(Universe):
    ASSETS = (
        {
            "ticker": "XLC",
            "assetclass": "Equity",
            "name": "Communication Services Select Sector SPDR Fund",
        },
        {
            "ticker": "XLY",
            "assetclass": "Equity",
            "name": "Consumer Discretionary Select Sector SPDR Fund",
        },
        {
            "ticker": "XLP",
            "assetclass": "Equity",
            "name": "Consumer Staples Select Sector SPDR Fund",
        },
        {
            "ticker": "XLE",
            "assetclass": "Equity",
            "name": "Energy Select Sector SPDR Fund",
        },
        {
            "ticker": "XLF",
            "assetclass": "Equity",
            "name": "Financial Select Sector SPDR Fund",
        },
        {
            "ticker": "XLV",
            "assetclass": "Equity",
            "name": "Health Care Select Sector SPDR Fund",
        },
        {
            "ticker": "XLI",
            "assetclass": "Equity",
            "name": "Industrial Select Sector SPDR Fund",
        },
        {
            "ticker": "XLB",
            "assetclass": "Equity",
            "name": "Materials Select Sector SPDR Fund",
        },
        {
            "ticker": "XLRE",
            "assetclass": "Equity",
            "name": "Real Estate Select Sector SPDR Fund",
        },
        {
            "ticker": "XLK",
            "assetclass": "Equity",
            "name": "Technology Select Sector SPDR Fund",
        },
        {
            "ticker": "XLU",
            "assetclass": "Equity",
            "name": "Utilities Select Sector SPDR Fund",
        },
    )


class UsAllocation(Universe):
    ASSETS = (
        {
            "ticker": "SPY",
            "assetclass": "Equity",
            "name": "US SPY Equity"
        },
        {
            "ticker": "AGG",
            "assetclass": "FixedIncome",
            "name": "US Aggregate Bond"
        },
    )

class GlobalAllocation(Universe):
    ASSETS = (
        {
            "ticker": "ACWI",
            "assetclass": "Equity",
            "name": "MSCI All Country"
        },
        {
            "ticker": "BND",
            "assetclass": "FixedIncome",
            "name": "Total Bond Market"
        },
    )
