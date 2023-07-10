from typing import Optional
import uuid
import pandas as pd
import streamlit as st
from src.core import universes
from .strategy_button import strategy_button

class Repeatable:
    cons = ["leverage", "weight", "return", "volatility"]

    def __init__(
        self,
        universe: Optional[universes.Universe] = None,
        prefix: str = "",
    ) -> None:
        self.prefix = prefix
        self.portfolio_constraint = {}
        self.asset_constraint = []
        if f"{self.prefix}_rows" not in st.session_state:
            st.session_state[f"{self.prefix}_rows"] = []
        self.universe = universe

    def add_row(self, name: str) -> None:
        element_id = uuid.uuid4()
        st.session_state[f"{self.prefix}_rows"].append(name + str(element_id))

    def del_row(self, row_id):
        st.session_state[f"{self.prefix}_rows"].remove(str(row_id))


    def generate_row(self, row):
        row_container = st.empty()
        row_columns = row_container.columns((3, 2, 2, 1))
        row_name = str(
            row_columns[0].selectbox(
                label="dd",
                options=self.cons,
                key=f"{self.prefix}_txt_{row}",
                label_visibility="collapsed",
            )
        )
        min_value = row_columns[1].number_input(
            label="Value",
            key=f"{self.prefix}_min_val_{row}",
            value=0,
            label_visibility="collapsed",
        )
        max_value = row_columns[2].number_input(
            label="Value",
            value=100,
            key=f"{self.prefix}_max_val_{row}",
            label_visibility="collapsed",
        )
        row_columns[3].button(
            label="🗑️",
            key=f"{self.prefix}_del_{row}",
            on_click=self.del_row,
            kwargs={"row_id": row},
        )
        return {f"min_{row_name}": min_value / 100, f"max_{row_name}": max_value / 100}

    def generate_assetclass_row(self, row):
        row_container = st.empty()
        row_columns = row_container.columns((3, 2, 2, 1))
        assetclasses = pd.DataFrame(self.universe.ASSETS).assetclass.unique()
        name = str(
            row_columns[0].selectbox(
                label="dd",
                options=assetclasses,
                key=f"{self.prefix}_txt_{row}",
                label_visibility="collapsed",
            )
        )
        min_value = row_columns[1].number_input(
            label="Value",
            key=f"{self.prefix}_min_val_{row}",
            value=0,
            label_visibility="collapsed",
        )
        max_value = row_columns[2].number_input(
            label="Value",
            value=100,
            key=f"{self.prefix}_max_val_{row}",
            label_visibility="collapsed",
        )
        row_columns[3].button(
            label="🗑️",
            key=f"{self.prefix}_del_{row}",
            on_click=self.del_row,
            kwargs={"row_id": row},
        )
        uni = pd.DataFrame(self.universe.ASSETS)
        return {
            "assets": list(uni[uni["assetclass"] == name]["ticker"]),
            "bounds": (min_value, max_value),
        }

    def generate_asset_row(self, row):
        row_container = st.empty()
        row_columns = row_container.columns((3, 2, 2, 1))
        uni = pd.DataFrame(self.universe.ASSETS)
        name = str(
            row_columns[0].selectbox(
                label="dd",
                options=list(uni["ticker"]),
                key=f"{self.prefix}_txt_{row}",
                label_visibility="collapsed",
            )
        )
        min_value = row_columns[1].number_input(
            label="Value",
            key=f"{self.prefix}_min_val_{row}",
            value=0,
            label_visibility="collapsed",
        )
        max_value = row_columns[2].number_input(
            label="Value",
            value=100,
            key=f"{self.prefix}_max_val_{row}",
            label_visibility="collapsed",
        )
        row_columns[3].button(
            label="🗑️",
            key=f"{self.prefix}_del_{row}",
            on_click=self.del_row,
            kwargs={"row_id": row},
        )

        return {
            "assets": [name],
            "bounds": (min_value, max_value),
        }

    def fit(self):
        if self.universe is not None:
            names = ["general", "assetclass", "asset"]

            for col, name in zip(st.columns(len(names)), names):
                with col:
                    st.button(
                        label=f"➕ {name.title()}",
                        key=f"{self.prefix}_{name}_add_constraint",
                        on_click=self.add_row,
                        kwargs={"name": name},
                    )
        else:
            name = "general"
            st.button(
                label="➕ Constraint",
                key=f"{self.prefix}_{name}_add_constraint",
                on_click=self.add_row,
                kwargs={"name": name},
            )



        for row in st.session_state[f"{self.prefix}_rows"]:
            if str(row).startswith("general"):
                row_data = self.generate_row(row)
                self.portfolio_constraint.update(row_data)
            elif str(row).startswith("assetclass"):
                row_data = self.generate_assetclass_row(row)
                self.asset_constraint.append(row_data)
            elif str(row).startswith("asset"):
                row_data = self.generate_asset_row(row)
                self.asset_constraint.append(row_data)

        return {
            "portfolio_constraint": self.portfolio_constraint,
            "asset_constraint": self.asset_constraint,
        }
