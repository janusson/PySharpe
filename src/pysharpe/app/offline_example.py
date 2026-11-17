"""A small, isolated example for exploring the dashboard without market data."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def _example_prices() -> pd.DataFrame:
    """Return fictional prices, not observations of actual securities."""

    return pd.DataFrame(
        {
            "Fictional ETF A": [100.0, 102.0, 99.0, 105.0, 103.0, 108.0, 106.0, 110.0],
            "Fictional ETF B": [100.0, 99.0, 101.0, 98.0, 102.0, 100.0, 104.0, 105.0],
        },
        index=pd.DatetimeIndex(
            [
                "2024-01-02",
                "2024-02-01",
                "2024-03-01",
                "2024-04-01",
                "2024-05-01",
                "2024-06-03",
                "2024-07-02",
                "2024-08-01",
            ],
            name="Date",
        ),
    )


def _price_multipliers(prices: pd.DataFrame) -> pd.DataFrame:
    """Compare complete, positive example prices with their first row."""

    return prices.div(prices.iloc[0])


def render_offline_example() -> None:
    """Show a synthetic, offline price-inspection walkthrough."""

    st.subheader("Explore a portfolio dataset, without the setup")
    st.write(
        "PySharpe is a research workspace for Canadian investors exploring "
        "broad-market ETFs. Start by inspecting two fictional price series; "
        "no account, personal holdings, or market-data connection is needed."
    )
    st.info(
        "Illustration only: the funds, dates, and prices are made up. "
        "This is not market history, a recommendation, or a prediction."
    )

    if st.button("Load example dataset", type="primary", key="load_offline_example"):
        st.session_state["offline_example_loaded"] = True
    if not st.session_state.get("offline_example_loaded", False):
        st.caption(
            "Select Load example dataset to see the data and a simple comparison."
        )
        return

    prices = _example_prices()
    st.divider()
    st.subheader("1 · The example data")
    st.write(
        "Two fictional CAD-denominated ETFs, shown at eight selected dates from "
        "January 2 to August 1, 2024. Each begins at CAD $100 per unit. "
        "The names are placeholders, not tradable ticker symbols."
    )

    with st.expander("View the fictional prices (CAD per unit)", expanded=False):
        st.dataframe(prices, width="stretch")
    st.download_button(
        "Download example prices CSV",
        data=prices.to_csv(index_label="Date").encode("utf-8"),
        file_name="pysharpe_fictional_prices.csv",
        mime="text/csv",
    )

    st.subheader("2 · Compare price movement")
    st.caption(
        "Each line is the price divided by its January 2 starting price: "
        "1.00× means unchanged. Both lines start at the 1.00× baseline. "
        "Lines connect only the example dates."
    )
    multipliers = _price_multipliers(prices)
    st.line_chart(multipliers, height=320, width="stretch")
    changes = multipliers.iloc[-1] - 1.0
    first, second = st.columns(2)
    with first:
        st.metric("Fictional ETF A · price change", f"{changes.iloc[0]:+.1%}")
    with second:
        st.metric("Fictional ETF B · price change", f"{changes.iloc[1]:+.1%}")
    st.caption(
        "Price changes only—not total returns or a portfolio result. "
        "Distributions, fees, taxes, withholding, FX, and trading are not modeled."
    )

    st.subheader("3 · Where to go next")
    st.write(
        "Download the example CSV to inspect the inputs. To explore the wider "
        "research dashboard, run the command below in a separate terminal. "
        "It can request market data immediately; the example is not "
        "automatically applied to it."
    )
    st.code("uv run streamlit run app.py", language="bash")
    st.caption(
        "The research workspace is a prototype. Its historical data and "
        "backtests have unresolved limitations; review their assumptions "
        "before interpreting results."
    )
