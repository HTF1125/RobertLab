import os
import streamlit.components.v1 as components

strategy_button = components.declare_component(
    name="strategy_button",
    path=os.path.dirname(os.path.abspath(__file__)),
)
