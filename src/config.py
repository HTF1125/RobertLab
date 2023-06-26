"""ROBERT"""

from typing import Optional

__all__ = ("Settings",)


class BaseSettings:
    platform: Optional[str] = None


Settings = BaseSettings()
