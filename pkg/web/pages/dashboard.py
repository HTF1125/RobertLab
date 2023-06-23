"""ROBERT"""
import pandas as pd

import streamlit as st
from .base import BasePage
from pkg.src.core import strategies


class Dashboard(BasePage):
    """Dashboard"""

    def load_page(self):
        st.info("Welcome to Robert's Dashboard.")

        li = {
            "UsSectorEwVcv1m": {
                "universe": "UnitedStatesSectors",
                "benchmark": "Global64",
                "inception": "2003-1-1",
                "frequency": "M",
                "commission": 10,
                "optimizer": "EqualWeight",
                "min_window": 252,
                "factors": ("VolumeCoefficientOfVariation1M",),
                "allow_fractional_shares": False,
            },
            "UsSectorEwPm6M1M": {
                "universe": "UnitedStatesSectors",
                "benchmark": "Global64",
                "inception": "2003-1-1",
                "frequency": "M",
                "commission": 10,
                "optimizer": "EqualWeight",
                "min_window": 252,
                "factors": ("PriceMomentum6M1M",),
                "allow_fractional_shares": False,
            },
            "GaaEwPm6M1M": {
                "universe": "GlobalAssetAllocation",
                "benchmark": "Global64",
                "inception": "2003-1-1",
                "frequency": "M",
                "commission": 10,
                "optimizer": "EqualWeight",
                "min_window": 252,
                "factors": ("PriceMomentum6M1M",),
                "allow_fractional_shares": False,
            },
            "GaaEwPm12M1M": {
                "universe": "GlobalAssetAllocation",
                "benchmark": "Global64",
                "inception": "2003-1-1",
                "frequency": "M",
                "commission": 10,
                "optimizer": "EqualWeight",
                "min_window": 252,
                "factors": ("PriceMomentum12M1M",),
                "allow_fractional_shares": False,
            },
            "GaaMinCorr": {
                "universe": "GlobalAssetAllocation",
                "benchmark": "Global64",
                "inception": "2003-1-1",
                "frequency": "M",
                "commission": 10,
                "optimizer": "MinCorrelation",
                "min_window": 252,
                "allow_fractional_shares": False,
            },
        }

        multistrategy = strategies.MultiStrategy()

        for name, strategy_load in li.items():
            with st.spinner(f"load strategy {name}"):
                strategy = multistrategy.load(name=name, **strategy_load)
                strategy.save(name)

        st.write(multistrategy.analytics.T)
        st.plotly_chart(
            self.line(multistrategy.performance.resample("M").last() / 10000),
            use_container_width=True,
        )
