import streamlit as st

# import database as db


st.set_page_config(
    page_title="ROBERT'S WEBSITE",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)


st.write("Robert's Website")


# with db.Session() as session:
#     st.multiselect(
#         label="Select investable assets",
#         options=[record for record in session.query(db.Investable).limit(10).all()],
#         format_func=lambda record: f"{record.name} ({record.ticker})",
#     )





with st.form("my_form"):
    st.write("Inside the form")
    slider_val = st.slider("Form slider")
    checkbox_val = st.checkbox("Form checkbox")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("slider", slider_val, "checkbox", checkbox_val)

st.write("Outside the form")



# Insert containers separated into tabs:
tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
tab1.write("this is tab 1")
tab2.write("this is tab 2")

# You can also use "with" notation:
with tab1:
    st.radio("Select one:", [1, 2])