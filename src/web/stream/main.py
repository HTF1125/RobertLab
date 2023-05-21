import streamlit as st


import database as db


st.write("test")


with db.Session() as session:
    st.multiselect(
        label="Select investable assets",
        options=[record for record in session.query(db.Investable).limit(10).all()],
        format_func=lambda record: f"{record.name} ({record.ticker})",
    )
