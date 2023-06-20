"""ROBERT"""
import os
import streamlit as st
from streamlit_option_menu import option_menu
from pkg.src.web import pages
from pkg.src import web


def load_css():
    file = os.path.join(os.path.dirname(web.__file__), "css", "base.css")
    with open(file=file, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init():
    st.set_page_config(
        page_title="ROBERT'S WEBSITE",
        page_icon="	:snowflake:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None,
    )

    load_css()


    with st.sidebar:
        selected = option_menu(
            menu_title="ROBERT",
            menu_icon="list-stars",
            options=[
                "Dashboard",
                "Market Regime",
                "Efficient Frontier",
                "Investment Strategy",
                "Alpha Factors",
                "About Me",
            ],
            orientation="vertical",
            styles={
                "container": {"padding": "5!important", "max-width": "100%"},
                "icon": {"font-size": "16px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "5",
                },
            },
        )

    getattr(pages, selected.replace(" ", ""))()
