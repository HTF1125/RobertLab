"""ROBERT"""
import streamlit as st
from .. import components, data, state
from pkg.src.core import feature, metrics


def get_features_percentile():

    c1, c2 = st.columns([5, 1])
    with c1:
        selected_features = st.multiselect(

            label="Select Features",
            options=[
                func for func in dir(feature)
                if callable(getattr(feature, func))
            ],
            format_func=lambda x: "".join(
                word.capitalize() for word in x.split("_")
            ),
        )
    with c2:
        percentile = (
            int(
                st.number_input(
                    label="Factor Percentile",
                    min_value=0,
                    max_value=100,
                    step=5,
                    value=70,
                )
            )
            / 100
        )

    return selected_features, percentile


def main():
    universe = components.get_universe()

    prices = data.get_prices(tickers=universe.ticker.tolist())


    with st.expander(label="See universe details:"):
        st.markdown("Universe:")
        st.dataframe(universe, height=150, use_container_width=True)
        st.markdown("Prices:")
        st.dataframe(prices, height=150, use_container_width=True)

    with st.form("AssetAllocationForm"):
        basic_calls = [
            components.get_name,
            components.get_start,
            components.get_end,
            components.get_objective,
            components.get_frequency,
            components.get_commission,
        ]
        basic_cols = st.columns([1] * len(basic_calls))

        basic_kwargs = {}

        for col, call in zip(basic_cols, basic_calls):
            with col:
                basic_kwargs[call.__name__[4:]] = call()

        selected_features, percentile = get_features_percentile()


        with st.expander("Asset Class Constraints"):
            asset_class_constraints = []

            for asset_class_in_universe in universe.assetclass.unique():
                asset_class_constraints.append(
                    {
                        "ticker": universe[
                            universe.assetclass == asset_class_in_universe
                        ].ticker.to_list(),
                        "bounds": st.select_slider(
                            label=f"{asset_class_in_universe}",
                            options=range(0, 105, 5),
                            value=(0, 100),
                        ),
                    }
                )

        with st.expander("Asset Constraints"):
            constraints = {}

            for _, asset in universe.iterrows():
                constraints[asset["ticker"]] = st.select_slider(
                    label=f"{asset['name']} ({asset['ticker']}))",
                    options=range(0, 105, 5),
                    value=(0, 100),
                )

        submitted = st.form_submit_button(label="Backtest", type="primary")

        if submitted:

            if selected_features:
                features = data.get_factors(
                    *selected_features, tickers=universe.ticker.tolist(),
                )
            else:
                features = None


            with st.spinner(text="Backtesting in progress..."):
                state.strategy.get_backtestmanager().Base(
                    **basic_kwargs,
                    prices=prices,
                    features=features,
                    percentile=percentile,
                )

    st.button(
        key="ClearStrategy",
        label="Clear All Strategies",
        on_click=state.strategy.get_backtestmanager().reset_strategies,
    )
    components.performances.main()
