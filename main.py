import streamlit as st
from plotly.graph_objects import Scatter
from plotly.subplots import make_subplots
from pkg.src.core.strategies import BacktestManager
from pkg.src.core import metrics
from web import components

st.set_page_config(
    page_title="ROBERT'S WEBSITE",
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f440.png",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

			html, body, [class*="css"]  {
			font-family: 'Roboto', sans-serif;
			}
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)


hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


def get_backtestmanager() -> BacktestManager:
    # Initialization
    if "backtestmanager" not in st.session_state:
        st.session_state["backtestmanager"] = BacktestManager()
    return st.session_state["backtestmanager"]


st.session_state["universe"] = st.selectbox(
    label="Select Investment Universe",
    options=["USSECTORETF", "General"],
)


get_backtestmanager().set_universe(name=st.session_state["universe"])


def clear_strategies():
    get_backtestmanager().reset_strategies()


momentum_tab, base_tab = st.tabs(["Momentum", "Base"])

with momentum_tab:
    st.button(label="Clear Strategies", on_click=clear_strategies)

    with st.form(key="momentum_month"):

        (
            objective,
            start,
            end,
            frequency,
            commission,
        ) = components.get_strategy_general_params()

        cols = st.columns([1] * 3)

        months = cols[0].select_slider(
            label="Momentum Months", options=range(1, 36 + 1), value=1
        )
        skip_months = cols[1].select_slider(
            label="Momentum Skip Months", options=range(0, 6 + 1), value=0
        )

        target_percentile = cols[2].select_slider(
            label="Target Percentile",
            options=range(0, 100 + 10, 10),
            value=70,
        )

        absolute = st.checkbox(label="Absolute Momentum", value=False)

        submitted = st.form_submit_button("Submit")
        if submitted:
            get_backtestmanager().commission = int(commission)
            get_backtestmanager().start = str(start)
            get_backtestmanager().end = str(end)
            get_backtestmanager().frequency = frequency
            with st.spinner(text="Backtesting in progress..."):

                get_backtestmanager().Momentum(
                    months=months,
                    skip_months=skip_months,
                    objective=objective,
                    absolute=absolute,
                    target_percentile=target_percentile / 100,
                )


if not get_backtestmanager().values.empty:

    st.write(get_backtestmanager().analytics.T)

    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
    )
    for name, strategy in get_backtestmanager().strategies.items():
        # Add line chart for prices to the first subplot
        val = strategy.value.resample("M").last()
        price_trace = Scatter(
            x=val.index,
            y=val.values,
            name=name,
            hovertemplate="Date: %{x}<br>Price: %{y}",
        )
        fig.add_trace(price_trace)

    fig.update_layout(
        title="Performance",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x",
        legend=dict(orientation="v", yanchor="top", y=1.1, xanchor="left", x=0),
    )

    fig.update_layout(
        xaxis=dict(title="Date", tickformat="%Y-%m-%d"),  # Customize the date format
        yaxis=dict(
            title="Price",
            tickprefix="$",  # Add a currency symbol to the y-axis tick labels
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


for name, strategy in get_backtestmanager().strategies.items():
    with st.expander(label=name, expanded=False):
        st.button(
            label="Delete",
            key=name,
            on_click=get_backtestmanager().drop_strategy,
            kwargs={"name": name},
        )
        fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
        )
        price_trace = Scatter(
            x=strategy.value.index,
            y=strategy.value.values,
            name="Strategy",
            hovertemplate="Date: %{x}<br>Price: %{y}",
        )
        price_bm_trace = Scatter(
            x=strategy.prices_bm.index,
            y=strategy.prices_bm.values,
            name="Benchmark",
            hovertemplate="Date: %{x}<br>Price: %{y}",
        )
        fig.add_trace(price_trace, row=1, col=1)
        fig.add_trace(price_bm_trace, row=1, col=1)
        # strategy_dd = metrics.to_drawdown(strategy.value)
        # benchmark_dd = metrics.to_drawdown(strategy.prices_bm)
        # price_trace = Scatter(
        #     x=strategy_dd.index,
        #     y=strategy_dd.values,
        #     name="Strategy",
        #     hovertemplate="Date: %{x}<br>Price: %{y}",
        # )
        # price_bm_trace = Scatter(
        #     x=benchmark_dd.index,
        #     y=benchmark_dd.values,
        #     name="Benchmark",
        #     hovertemplate="Date: %{x}<br>Price: %{y}",
        # )

        # fig.add_trace(strategy_dd, row=2, col=1)
        # fig.add_trace(benchmark_dd, row=2, col=1)
        fig.update_layout(
            title="Performance",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x",
            legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0),
        )

        fig.update_layout(
            xaxis=dict(
                title="Date", tickformat="%Y-%m-%d"
            ),  # Customize the date format
            yaxis=dict(
                title="Price",
                tickprefix="$",  # Add a currency symbol to the y-axis tick labels
            ),
        )

        st.plotly_chart(fig, use_container_width=True)



m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ce1126;
    color: white;
    height: 2em;
    border-radius:10px;
    border:3px solid #000000;
    font-size:20px;
    font-weight: bold;
    margin: auto;
    display: block;
}

div.stButton > button:hover {
	background:linear-gradient(to bottom, #ce1126 5%, #ff5a5a 100%);
	background-color:#ce1126;
}

div.stButton > button:active {
	position:relative;
	top:3px;
}

</style>""", unsafe_allow_html=True)

