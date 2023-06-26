"""ROBERT"""

import streamlit as st
from src.core import MultiStrategy
from .base import BasePage


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
        import plotly.graph_objects as go

        multistrategy = MultiStrategy()
        fig = go.Figure()

        with st.spinner(f"load strategies..."):
            strategy = multistrategy.from_files()

        for name, strategy in multistrategy.items():
            fig.add_trace(
                go.Scatter(
                    x=strategy.performance.index,
                    y=strategy.performance.values,
                    name=name,
                )
            )

        fig.update_layout(
            title="Performance", legend_orientation="h", hovermode="x unified"
        )

        st.write(multistrategy.analytics.T)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )
