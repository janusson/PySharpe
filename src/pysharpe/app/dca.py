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
    final_bal = projection.final_balance()
    total_contrib = projection.final_contribution()
    net_gain = final_bal - total_contrib
    gain_pct = (net_gain / total_contrib * 100) if total_contrib > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Final Balance",
            f"${final_bal:,.2f}",
            help="Total projected value of your portfolio at the end of the simulation period.",
        )
    with col2:
        st.metric(
            "Total Contributions",
            f"${total_contrib:,.2f}",
            help="Total cumulative cash deposited over the entire investment horizon.",
        )
    with col3:
        st.metric(
            "Net Investment Gain",
            f"${net_gain:,.2f}",
            delta=f"{gain_pct:+.2f}%",
            help="Total profit generated from compounding returns above your cumulative deposits.",
        )
    return df


__all__ = ["render_dca_projection"]
