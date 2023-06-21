import streamlit as st
from .base import BasePage


class Dashboard(BasePage):
    def render(self):
        return st.info("Welcome to Robert's Dashboard.")
