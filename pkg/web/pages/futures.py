"""ROBERT"""
import streamlit as st
from .base import BasePage


class Futures(BasePage):
    """
    This sections holds all futures development prototypes.
    """

    def load_page(self):
        st.write(
            """
            Development Plan:

            `Benchmarks`:
            1. Build class.
            2. Incorporate with portfolio optimizer.

            `Strategies`:
            1. Build Plot class.
            2. Make in class plot functions with Plot class.

            """
        )