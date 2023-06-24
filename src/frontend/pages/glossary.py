"""ROBERT"""
import streamlit as st
from .base import BasePage


class Glossary(BasePage):
    def load_page(self):
        terms = {
            "investment universe": {
                "definition": """
                    The term "investment universe" refers to the entire set of
                    investment options or securities that an investor or
                    investment manager considers when making investment
                    decisions.
                    It encompasses all the available investment opportunities
                    within a given context, such as a particular market, asset
                    class, or investment strategy.\n
                    The investment universe can vary depending on the specific
                    goals, strategies, and constraints of the investor or
                    investment manager. For example:\n
                    1. Asset Class: The investment universe may focus on a
                    specific asset class, such as equities (stocks), fixed
                    income (bonds), real estate, commodities, or alternative
                    investments.
                    2. Geographic Region: The investment universe may be
                    defined by a particular geographic region, such as
                    investments limited to a specific country, region, or
                    global markets.
                    3. Market Cap: The investment universe may be limited to
                    specific market capitalization ranges, such as large-cap,
                    mid-cap, small-cap, or a combination of these.
                    4. Investment Style: The investment universe may be defined
                    by an investment style, such as value investing, growth
                    investing, income-focused, or socially responsible investing.
                    5. Sector or Industry: The investment universe may focus on
                    specific sectors or industries, such as technology,
                    healthcare, energy, or consumer goods.\n
                    The investment universe serves as a starting point for
                    investment analysis and decision-making. It provides a
                    framework for evaluating and selecting investment options
                    that align with the investor's objectives, risk tolerance,
                    and investment strategy.\n
                    It's important to note that the investment universe is not
                    static and can evolve over time as market conditions change,
                    new investment opportunities arise, or investment
                    preferences and strategies are adjusted.
                    """
            },
            "investment time horizon" : {
                "definition" : """
                    time horizon is when you think your trade is going to play out.

                    Since the length of time it takes for a investment to reach
                    the target, it often goes a very volatile path. Therefore,
                    if it is a long-term trend you tend to put less month into
                    it, because the market may have large swings to capture the
                    upside target.
                    And investors should size their position accordingly.

                """
            }
        }

        searched = st.multiselect(label="Select Terms", options=terms.keys())
        st.write(searched)

        self.write(terms)

    def write(self, terms):
        for name, term in terms.items():
            st.write(name)
            if isinstance(term, dict):
                self.write(term)
            else:
                st.write(term)
