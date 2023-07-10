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
<<<<<<< HEAD
from .market_regime import MarketRegime
from .base import BasePage
=======
from .regime import Regime
>>>>>>> 33e7afd09be2eaae3279fd2181eea2bb79a48b85

__all__ = [
    "Dashboard",
    "GlobalMacro",
<<<<<<< HEAD
    "MarketRegime",
    "CapitalMarket",
=======
    "Regime",
    "EfficientFrontier",
>>>>>>> 33e7afd09be2eaae3279fd2181eea2bb79a48b85
    "AlphaFactors",
    "MultiStrategy",
    "AboutMe",
    "Glossary",
    "Futures",
]


def get(page: str) -> BasePage:
    # Use getattr() to get the attribute value
    return getattr(sys.modules[__name__], page)()
