"""ROBERT"""
import pandas as pd
import streamlit as st
from pkg.src.core import metrics
from .base import BasePage
from .. import data


class MarketRegime(BasePage):
    def render(self):
        with st.form(key="Market Regime Form"):
            start, end = self.get_date_range()
            submitted = st.form_submit_button("Submit")
            if submitted:
                self.get_vix_regime(start=start, end=end)
                self.get_oecd_us_lei_regime(start=start, end=end)
                yield_data = self.inflation_short_yield_data()
                pp = (
                    metrics.to_macd(yield_data).rolling(121).corr().unstack().iloc[:, 0]
                    + 1
                )
                pp = pp.clip(0, 1)
                pp.name = "fff"
                st.plotly_chart(self.pie(pp.to_frame().dropna()), use_container_width=True)

    def get_vix_regime(self, start, end):
        vix = data.get_vix()
        normalized = metrics.rolling.to_standard_scalar(vix, window=252)
        normalized = normalized.ewm(90).mean()
        normalized = normalized.clip(lower=-3, upper=3)
        st.plotly_chart(
            self.bar(data=normalized.loc[start:end]),
            use_container_width=True,
        )

    def get_oecd_us_lei_regime(self, start, end):
        lei = data.get_oecd_us_lei()
        lei.index = lei.index + pd.DateOffset(months=1)
        change = lei.resample("M").last().diff().dropna()
        normalized = metrics.rolling.to_standard_scalar(change, window=12 * 5)
        normalized = normalized.clip(lower=-3, upper=3)
        st.plotly_chart(
            self.bar(data=normalized.dropna().loc[start:end]),
            use_container_width=True,
        )

    def inflation_short_yield_data(self, start=None, end=None) -> pd.DataFrame:
        """
        Get raw data for leading economic indicator regime.
        """
        tickers = dict(THREEFFTP10="term", T10YIE="inflation", DGS10="treasury")

        start = start if start else "1900-01-01"
        import pandas_datareader as pdr

        dt = pdr.DataReader(list(tickers.keys()), "fred", start=start)
        dt = dt.rename(columns=tickers)
        dt["short_yield"] = dt["treasury"] - dt["term"] - dt["inflation"]
        dt = dt[["short_yield", "inflation"]]
        return dt.dropna()
