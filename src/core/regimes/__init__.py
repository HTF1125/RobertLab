import sys
from typing import Type, Union
from .base import Regime
from .ext import *

__all__ = ["OneRegime", "VolatilityRegime", "UsLeiRegime"]


def get(regime: Union[str, Regime, Type[Regime]]) -> Regime:
    # Use getattr() to get the attribute value
    try:
        if isinstance(regime, str):
            return getattr(sys.modules[__name__], regime)()
        if isinstance(regime, type) and issubclass(regime, Regime):
            return regime()
        if issubclass(regime.__class__, Regime):
            return regime
        return getattr(sys.modules[__name__], str(regime))()
    except AttributeError as exc:
        raise ValueError(f"Invalid factor: {regime}") from exc
