"""ROBERT"""
import sys
from .about_robert import AboutMe
from .global_macro import GlobalMacro
from .alpha_factors import AlphaFactors
from .dashboard import Dashboard
from .multistrategy import MultiStrategy
from .capital_market import CapitalMarket
from .glossary import Glossary
from .futures import Futures
from .market_regime import MarketRegime
from .base import BasePage

__all__ = [
    "Dashboard",
    "GlobalMacro",
    "MarketRegime",
    "CapitalMarket",
    "AlphaFactors",
    "MultiStrategy",
    "AboutMe",
    "Glossary",
    "Futures",
]


def get(page: str) -> BasePage:
    # Use getattr() to get the attribute value
    return getattr(sys.modules[__name__], page)()
