"""ROBERT"""

import streamlit as st
from streamlit_option_menu import option_menu
from pkg.src.web import components
from pkg.src.web.pages import asset_allocation

st.set_page_config(
    page_title="ROBERT'S WEBSITE",
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f440.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None,
)

# utils.local_css("web/css/base.css")

selected = option_menu(
    menu_title="Robert",
    menu_icon="diagram-3",
    options=["Dashboard", "Strategy", "AssetAllocation"],
    # icons=['house', 'stars'],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "5", "max-width": "100%"},
        "icon": {"font-size": "16px"},
        "nav-link": {
            "font-size": "15px",
            "text-align": "middle",
            "margin": "0px",
            "--hover-color": "#eee",
        },
    },
)


if selected == "Strategy":
    components.momentum.main()
    components.performances.main()

if selected == "Dashboard":
    start, end = components.get_date_range()
    left, right = st.columns([1, 1])
    with left:
        components.macro.make_yield_curve()
        components.macro.make_yield_curve2()
    with right:
        components.macro.make_inflation_linked()



if selected == "AssetAllocation":
    asset_allocation.main()