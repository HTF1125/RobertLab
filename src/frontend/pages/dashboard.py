"""ROBERT"""

import streamlit as st
from .base import BasePage

class Dashboard(BasePage):
    """Dashboard"""

    def load_page(self):
        st.info("Welcome to Robert's Dashboard.")
