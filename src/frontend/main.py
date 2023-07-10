import streamlit as st
from streamlit_option_menu import option_menu
from src.frontend import pages
from src.backend import config
from . import static
from . import session


def init():
    st.set_page_config(
        page_title="ROBERT'S WEBSITE",
        page_icon=":snowflake:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None,
    )

    config.Settings.PLATFORM = "Streamlit"

    # Load HTML and CSS files
    filenames = static.all_filenames()
    for filename in filenames:
        with open(file=filename, encoding="utf-8") as f:
            content = f.read()
            if filename.endswith(".html"):
                st.markdown(body=content, unsafe_allow_html=True)
            else:
                st.markdown(body=f"<style>{content}</style>", unsafe_allow_html=True)

    # Set the current page
    page = session.getPage()

    # Sidebar
    with st.sidebar:
        st.markdown(
            "<h1 style='width: 100%; text-align: center;'>ROBERT'S DASHBOARD</h1>",
            unsafe_allow_html=True,
        )

        # Page selection
        page = option_menu(
            menu_title=None,
            options=pages.__all__,
            default_index=int(pages.__all__.index(page)),
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
        st.markdown(body="Version: 0.0.1")
        session.setPage(page=page, rerun=True)
    # Load the selected page
    pages.get(session.getPage()).load()
