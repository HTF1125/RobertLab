"""ROBERT"""
import uuid
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from src.core import universes
from .base import BasePage
from .. import components
from .. import session


class ConstraintSetter:
    def __init__(self, universe: universes.Universe, prefix: str = "") -> None:
        self.universe = universe
        self.prefix = prefix
        self.portfolio_constraint = {}
        self.asset_constraint = []
        if f"{self.prefix}_rows" not in st.session_state:
            st.session_state[f"{self.prefix}_rows"] = []

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


class MultiStrategy(BasePage):
    def load_page(self):
        multistrategy = session.getNewStrategy()

        col1, col2 = st.columns(2)
        with col1:
            universe = components.single.get_universe()
        with col2:
            regime = components.single.get_regime()
        constraint = {}
        if regime.__states__:
            with st.expander("Apply Constraints", expanded=False):
                for col, state in zip(
                    st.columns(len(regime.__states__)), regime.__states__
                ):
                    with col:
                        self.h4(state)
                        constraint[state] = ConstraintSetter(universe=universe, prefix=state).fit()

        with st.form("AssetAllocationForm"):
            strategy_params = components.single.get_strategy_parameters()

            submitted = st.form_submit_button(label="Backtest", type="primary")

            if submitted:
                with st.spinner(text="Backtesting in progress..."):
                    multistrategy.add_strategy(
                        universe=universe,
                        regime=regime,
                        constraint=constraint,
                        **strategy_params,
                    )
        if multistrategy:
            st.button(label="Clear Strategies", on_click=multistrategy.clear)

            fig = go.Figure()

            for name, strategy in multistrategy.items():
                performance = strategy.performance / strategy.principal - 1
                num_points = len(performance)
                indices = np.linspace(0, num_points - 1, 30, dtype=int)
                performance = performance.iloc[indices]
                fig.add_trace(
                    go.Scatter(
                        x=performance.index,
                        y=performance.values,
                        name=name,
                    )
                )

            fig.update_layout(
                yaxis_tickformat=".0%",
            )
            self.plotly(fig, title="Performance")
            st.table(multistrategy.analytics.T)

        components.plot_multistrategy(multistrategy, allow_save=True)
