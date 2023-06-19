"""ROBERT"""

import streamlit as st
from streamlit_option_menu import option_menu
from pkg.src.web import components, pages


def init():
    st.set_page_config(
        page_title="ROBERT'S WEBSITE",
        page_icon="	:snowflake:",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items=None,
    )

    with st.sidebar:
        selected = option_menu(
            menu_title="Welcome",
            menu_icon="list-stars",
            options=[
                "Dashboard",
                "Mkt.Regime",
                "Eff.Frontier",
                "Ivt.Strategy",
                "Alp.Factor",
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

    components.add_badges()

    st.subheader(selected)
    st.markdown(
        '<hr style="margin-top: 0px; margin-bottom: 5px;">', unsafe_allow_html=True
    )

    if selected == "Dashboard":
        st.info("welcome to robert's dashboard.")

    if selected == "Ivt.Strategy":
        pages.allocation.main()

    if selected == "Eff.Frontier":
        pages.efficient_frontier.main()

    if selected == "Mkt.Regime":
        pages.regime.main()

    if selected == "Alp.Factor":
        pass
