"""ROBERT"""
import streamlit as st
from streamlit_option_menu import option_menu
from src.frontend import pages


def init():
    st.set_page_config(
        page_title="ROBERT'S WEBSITE",
        page_icon="	:snowflake:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None,
    )

    with st.sidebar:
        st.markdown(
            "<h1 style='width: 100%;'>ROBERT'S DASHBOARD</h1>", unsafe_allow_html=True
        )

        selected = option_menu(
            menu_title=None,
            options=pages.__all__,
            orientation="vertical",
            styles={
                "container": {"padding": "10!important", "max-width": "100%"},
                "icon": {"font-size": "16px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "5px",
                },
            },
        )

    getattr(pages, selected)()
