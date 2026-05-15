"""Dollar-cost averaging helpers for the Streamlit app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pysharpe.visualization import simulate_dca


def render_dca_projection(
    months: int,
    initial: float,
    monthly: float,
    rate: float,
) -> pd.DataFrame:
    """Simulate and plot a dollar-cost averaging projection."""

    projection = simulate_dca(
        months=months,
        initial_investment=initial,
        monthly_contribution=monthly,
        annual_return_rate=rate,
    )
    df = pd.DataFrame(
        {
            "Months": projection.months,
            "Balance": projection.balances,
            "Contributions": projection.contributions,
        }
    )
    st.line_chart(df.set_index("Months"), height=320)
    st.metric("Final Balance", f"${projection.final_balance():,.2f}")
    st.metric("Total Contributions", f"${projection.final_contribution():,.2f}")
    return df


__all__ = ["render_dca_projection"]
