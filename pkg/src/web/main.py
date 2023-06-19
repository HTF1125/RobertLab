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
            options=["Dashboard", "Eff.Frontier", "Allocation"],
            orientation="vertical",
            styles={
                "container": {"padding": "5!important", "max-width": "100%"},
                "icon": {"font-size": "16px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "5",
                    # "--hover-color": "#eee",
                },
            },
        )

    tt = f"""
    [![GitHub](https://img.shields.io/badge/GitHub-htf1125-black?logo=github)](https://github.com/htf1125/RobertLab)
    [![GitHub](https://img.shields.io/github/license/htf1125/robertlab)](https://github.com/HTF1125/RobertLab)
    """
    st.markdown(tt, unsafe_allow_html=True)

    st.title(selected)

    # if selected == "Dashboard":
    #     start, end = components.get_date_range()
    #     left, right = st.columns([1, 1])
    #     with left:
    #         components.macro.make_yield_curve()
    #         components.macro.make_yield_curve2()
    #     with right:
    #         components.macro.make_inflation_linked()

    if selected == "Allocation":
        pages.allocation.main()

    if selected == "Eff.Frontier":
        pages.efficient_frontier.main()

    # if selected == "Regime":
    #     pages.regime.main()

    # if selected == "Security":
    #     pass


