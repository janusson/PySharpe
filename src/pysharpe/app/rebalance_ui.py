"""Execution/rebalance tab for the PySharpe Streamlit dashboard.

Provides an iPad-friendly interface for uploading broker-exported portfolio
CSVs, configuring execution parameters, and generating a RebalancePlan.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

try:
    import streamlit as st  # type: ignore[import]
except ImportError:  # pragma: no cover
    st = None  # type: ignore[assignment]

import shutil

from pysharpe.config import ExecutionConfig
from pysharpe.execution.rebalance import RebalancePlan, build_rebalance_plan


def _st_css() -> None:
    """Inject minimal CSS for the execution checklist cards."""
    st.markdown(
        """
        <style>
        .bx-card {
            background: #f1f8e9;
            border: 2px solid #4caf50;
            border-radius: 12px;
            padding: 16px 20px;
            margin: 8px 0;
        }
        .bx-card h4 { margin-top: 0; color: #1b5e20; }
        .leftover-banner {
            background: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 8px;
            padding: 10px 16px;
            margin: 8px 0;
            color: #0d47a1;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_money(value: float | None) -> str:
    """Format a dollar amount for display."""
    if value is None or pd.isna(value):
        return "—"
    return f"${float(value):,.2f}"


def _format_shares(value: float | None) -> str:
    """Format a share count for display."""
    if value is None or pd.isna(value):
        return "—"
    v = float(value)
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.4f}"


def _render_buy_orders_table(plan: RebalancePlan) -> None:
    """Display buy orders in a styled, iPad-friendly table."""
    buy = plan.buy_orders.copy()
    if buy.empty:
        st.info("No buy orders recommended. Your portfolio may already be at target.")
        return

    display_cols = {
        "ticker": "Ticker",
        "latest_price": "Price",
        "current_weight": "Current",
        "target_weight": "Target",
        "opportunity_score": "Score",
        "recommended_allocation": "Buy $",
        "recommended_shares": "Buy Shares",
    }
    available = [c for c in display_cols if c in buy.columns]
    display = buy[available].rename(columns=display_cols).copy()
    display["Price"] = display["Price"].apply(_format_money)
    display["Buy $"] = display["Buy $"].apply(_format_money)
    display["Buy Shares"] = display["Buy Shares"].apply(_format_shares)
    display["Current"] = display["Current"].apply(
        lambda v: f"{v:.2%}" if not pd.isna(v) else "—"
    )
    display["Target"] = display["Target"].apply(
        lambda v: f"{v:.2%}" if not pd.isna(v) else "—"
    )
    display["Score"] = display["Score"].apply(
        lambda v: f"{v:.4f}" if not pd.isna(v) else "—"
    )

    st.markdown("### 💰 Buy Orders")
    st.dataframe(display, use_container_width=True, hide_index=True)

    total_buy = buy["recommended_allocation"].sum()
    st.caption(f"Total deployment: **{_format_money(total_buy)}**")

    if "leftover_cash" in plan.allocations.columns:
        leftover = plan.allocations["leftover_cash"].iloc[0]
        if leftover and leftover > 0:
            st.markdown(
                f'<div class="leftover-banner">'
                f"⚠️ Unallocated (fractional rounding): {_format_money(leftover)}"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_account_breakdown(plan: RebalancePlan) -> None:
    """Display per-account allocations for multi-account plans."""
    if not plan.is_multi_account or plan.account_allocations is None:
        return

    st.markdown("---")
    st.markdown("### 🏦 Account Breakdown")

    for account in plan.accounts:
        acct_alloc = plan.account_allocations[account]
        acct_cash = plan.account_cash.get(account, 0.0) if plan.account_cash else 0.0

        buys = acct_alloc[acct_alloc["recommended_allocation"] > 0]
        if buys.empty:
            st.caption(f"**{account}** — No buys (${acct_cash:,.2f} allocated)")
            continue

        with st.expander(f"📁 {account} — ${acct_cash:,.2f}"):
            display = pd.DataFrame(
                {
                    "Ticker": buys["ticker"],
                    "Buy $": buys["recommended_allocation"].apply(_format_money),
                    "Buy Shares": buys["recommended_shares"].apply(_format_shares),
                    "Score": buys["opportunity_score"].apply(lambda v: f"{v:.4f}"),
                }
            )
            st.dataframe(display, use_container_width=True, hide_index=True)


def _derive_latest_prices(price_data: pd.DataFrame) -> pd.DataFrame:
    """Extract the most recent non-null prices from loaded price data.

    Returns a DataFrame with ``ticker`` and ``latest_price`` columns.
    """
    if price_data.empty:
        return pd.DataFrame(columns=["ticker", "latest_price"])

    numeric = price_data.select_dtypes("number")
    if numeric.empty:
        return pd.DataFrame(columns=["ticker", "latest_price"])

    latest = numeric.ffill().iloc[-1].dropna()
    result = latest.rename_axis("ticker").reset_index(name="latest_price")
    result["ticker"] = result["ticker"].astype(str).str.strip()
    return result


def render_execution_tab(
    price_data: pd.DataFrame,
    *,
    default_cash: float = 1000.0,
) -> None:
    """Render the Value Averaging execution tab.

    Parameters
    ----------
    price_data : pd.DataFrame
        Currently loaded price history (wide format: dates × tickers).
        Used to derive latest prices when no collated CSV is available.
    default_cash : float
        Default monthly contribution amount in dollars.
    """
    _st_css()

    st.subheader("📋 Monthly VA Execution")
    st.caption(
        "Upload your broker-exported portfolio CSV, set your contribution, "
        "and generate a buy plan — optimized for iPad execution."
    )

    # ── Row 1: Holdings + New Cash ──────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 1️⃣ Portfolio Holdings")
        holdings_file = st.file_uploader(
            "Upload broker-exported CSV",
            type=["csv"],
            help=(
                "CSV with columns like: Symbol/Ticker, Quantity/Shares or "
                "Market Value, and optionally Account. "
                "Most broker exports work directly."
            ),
            key="exec_holdings",
        )
        holdings_preview = None
        if holdings_file is not None:
            try:
                holdings_preview = pd.read_csv(holdings_file)
                holdings_file.seek(0)  # Reset for downstream use
                st.caption(
                    f"Loaded **{len(holdings_preview)}** positions "
                    f"({', '.join(str(c) for c in holdings_preview.columns)})"
                )
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")
                holdings_file = None

    with col2:
        st.markdown("#### 2️⃣ New Cash")
        new_cash = st.number_input(
            "Monthly contribution ($)",
            min_value=0.0,
            value=default_cash,
            step=100.0,
            format="%.2f",
            key="exec_new_cash",
        )

    # ── Row 2: Target Weights ───────────────────────────────────────────
    st.markdown("#### 3️⃣ Target Weights")
    weights_mode = st.radio(
        "Source",
        ["Pre-computed portfolio", "Upload CSV", "Equal weight"],
        horizontal=True,
        key="exec_weights_mode",
        help="Pre-computed: uses saved optimiser output. Upload: your own weights CSV.",
    )

    weights_df: pd.DataFrame | None = None
    portfolio_name: str = "streamlit"

    if weights_mode == "Pre-computed portfolio":
        from pysharpe import EXPORT_DIR

        export_root = Path(EXPORT_DIR).expanduser().resolve()
        available: list[str] = []
        if export_root.exists():
            for p in sorted(export_root.glob("*_weights.txt")):
                name = p.stem.replace("_weights", "")
                available.append(name)

        if available:
            portfolio_name = st.selectbox(
                "Portfolio",
                available,
                key="exec_portfolio",
                help="Portfolios found in the configured export directory.",
            )
            st.caption(
                f"Using weights from `{export_root / f'{portfolio_name}_weights.txt'}`"
            )
        else:
            st.warning(
                "No pre-computed portfolios found. "
                "Run `pysharpe optimise --portfolio <name>` first, "
                "or switch to Upload / Equal weight."
            )
            st.stop()

    elif weights_mode == "Upload CSV":
        weights_file = st.file_uploader(
            "Weights CSV (ticker, weight)",
            type=["csv"],
            key="exec_weights_file",
        )
        if weights_file is not None:
            try:
                weights_df = pd.read_csv(weights_file)
            except Exception as exc:
                st.error(f"Could not parse weights: {exc}")
                return

    # ── Price data status ──────────────────────────────────────────────
    has_price_data = (
        not price_data.empty and not price_data.select_dtypes("number").empty
    )
    if weights_mode != "Pre-computed portfolio" and not has_price_data:
        st.warning(
            "⚠️ No price data loaded. Either go to the **Analytics** tab "
            "to download prices, or upload a latest-prices CSV below."
        )
        prices_upload = st.file_uploader(
            "Upload latest prices CSV (ticker, latest_price)",
            type=["csv"],
            key="exec_prices_upload",
        )
        if prices_upload is not None:
            try:
                prices_df = pd.read_csv(prices_upload)
                if (
                    "ticker" in prices_df.columns
                    and "latest_price" in prices_df.columns
                ):
                    row_data = {
                        row["ticker"]: [row["latest_price"]]
                        for _, row in prices_df.iterrows()
                    }
                    price_data = pd.DataFrame(
                        row_data, index=pd.Index(["2024-01-01"], name="Date")
                    )
                    st.caption(f"Loaded {len(prices_df)} price points")
            except Exception as exc:
                st.error(f"Could not parse prices: {exc}")

    # ── Row 3: Execution Settings ───────────────────────────────────────
    with st.expander("⚙️ Execution Settings", expanded=False):
        cfg_col1, cfg_col2 = st.columns(2)

        with cfg_col1:
            account_type = st.selectbox(
                "Account type",
                ["TFSA", "RRSP", "FHSA", "Non-Reg"],
                index=0,
                key="exec_acct_type",
                help="TFSA/FHSA: 15% US withholding tax drag applied.",
            )
            allow_fractional = st.checkbox(
                "Allow fractional shares",
                value=False,
                key="exec_fractional",
                help="When disabled, shares are floored and leftover cash tracked.",
            )

        with cfg_col2:
            fx_fee_bps = st.number_input(
                "FX fee (bps)",
                min_value=0.0,
                value=150.0,
                step=25.0,
                key="exec_fx_fee",
                help="Broker FX conversion fee in basis points (150 = 1.5%).",
            )

        if fx_fee_bps > 0:
            cost_col1, _ = st.columns(2)
            with cost_col1:
                example_amount = 5000.0
                standard = example_amount * (fx_fee_bps / 10000)
                st.caption(
                    f"Example on ${example_amount:,.0f}: "
                    f"standard FX fee = ${standard:,.2f}"
                )

    # ── Row 4: Action ───────────────────────────────────────────────────
    st.markdown("---")

    weights_ready = weights_mode != "Upload CSV" or weights_df is not None
    can_run = holdings_file is not None and new_cash > 0 and weights_ready
    if not can_run:
        if holdings_file is None:
            st.info("Upload your portfolio holdings CSV to enable the rebalance plan.")
        if new_cash <= 0:
            st.info("Set a positive contribution amount.")
        if not weights_ready:
            st.info("Upload a weights CSV or switch to another weights source.")

    if st.button(
        "🚀 Run Rebalance Plan",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
        key="exec_run_button",
    ):
        _execute_rebalance(
            holdings_file=holdings_file,
            new_cash=new_cash,
            price_data=price_data,
            weights_df=weights_df,
            portfolio_name=portfolio_name,
            weights_mode=weights_mode,
            account_type=account_type,
            allow_fractional=allow_fractional,
            fx_fee_bps=fx_fee_bps,
        )


def _execute_rebalance(
    *,
    holdings_file,
    new_cash: float,
    price_data: pd.DataFrame,
    weights_df: pd.DataFrame | None,
    portfolio_name: str,
    weights_mode: str,
    account_type: str,
    allow_fractional: bool,
    fx_fee_bps: float,
) -> None:
    """Run build_rebalance_plan and render results."""
    # --- Build execution config ---
    execution_config = ExecutionConfig(
        account_type=account_type,
        allow_fractional=allow_fractional,
        fx_fee_bps=fx_fee_bps,
    )

    # --- Create temp export directory ---
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir)

        # Write weights file
        if weights_mode == "Pre-computed portfolio":
            from pysharpe import EXPORT_DIR

            existing_weights = Path(EXPORT_DIR) / f"{portfolio_name}_weights.txt"
            existing_collated = Path(EXPORT_DIR) / f"{portfolio_name}_collated.csv"
            if not existing_weights.exists():
                st.error(f"Weights file not found: {existing_weights}")
                return

            # Copy the pre-computed weights file
            shutil.copy(existing_weights, export_dir / f"{portfolio_name}_weights.txt")

            # Derive/copy latest prices
            if existing_collated.exists():
                shutil.copy(
                    existing_collated, export_dir / f"{portfolio_name}_collated.csv"
                )
            else:
                _write_collated_from_prices(price_data, export_dir, portfolio_name)
        elif weights_df is not None:
            weights_df.to_csv(export_dir / f"{portfolio_name}_weights.txt", index=False)
            _write_collated_from_prices(price_data, export_dir, portfolio_name)
        else:
            # Equal weight: derive from holdings
            holdings_file.seek(0)
            holdings_df = pd.read_csv(holdings_file)
            holdings_file.seek(0)  # Reset again for the write below
            from pysharpe.execution.rebalance import _rename_known_columns

            renamed = _rename_known_columns(holdings_df)
            tickers = (
                renamed["ticker"].dropna().unique().tolist()
                if "ticker" in renamed.columns
                else []
            )
            if not tickers:
                st.error("Could not identify ticker column in holdings CSV.")
                return

            n = len(tickers)
            equal_weight = 1.0 / n
            eq_df = pd.DataFrame({"ticker": tickers, "weight": [equal_weight] * n})
            eq_df.to_csv(export_dir / f"{portfolio_name}_weights.txt", index=False)
            _write_collated_from_prices(price_data, export_dir, portfolio_name)

        # --- Save holdings to temp file ---
        holdings_file.seek(0)
        holdings_path = export_dir / "holdings.csv"
        holdings_path.write_bytes(holdings_file.getvalue())

        # --- Run rebalance ---
        try:
            with st.spinner("Computing rebalance plan..."):
                plan = build_rebalance_plan(
                    portfolio_name=portfolio_name,
                    new_cash=new_cash,
                    holdings_csv=holdings_path,
                    export_dir=export_dir,
                    execution_config=execution_config,
                )
        except Exception as exc:
            st.error(f"Rebalance plan failed: {exc}")
            return

    # --- Render results ---
    st.markdown("---")

    # Summary metrics
    m1, m2 = st.columns(2)
    buy_count = len(plan.buy_orders)
    total_buy = plan.buy_orders["recommended_allocation"].sum()

    m1.metric("Buy Orders", str(buy_count))
    m2.metric("Total Deployment", _format_money(total_buy))

    # Buy orders
    _render_buy_orders_table(plan)

    # Account breakdown
    _render_account_breakdown(plan)

    # Download
    scored_csv = plan.scored_state.to_csv(index=False).encode("utf-8")
    alloc_csv = plan.allocations.to_csv(index=False).encode("utf-8")

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "📥 Download Full Plan (CSV)",
            data=alloc_csv,
            file_name=f"{portfolio_name}_rebalance_plan.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            "📊 Download Scored State (CSV)",
            data=scored_csv,
            file_name=f"{portfolio_name}_scored_state.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _write_collated_from_prices(
    price_data: pd.DataFrame,
    export_dir: Path,
    portfolio_name: str,
) -> None:
    """Save latest prices as a minimal collated CSV for build_rebalance_plan.

    When ``price_data`` is empty, writes a single-row placeholder so the
    upstream loader does not crash on missing-file errors.  Missing ticker
    warnings are deferred to ``build_rebalance_plan``.
    """
    latest = _derive_latest_prices(price_data)
    if latest.empty or "latest_price" not in latest.columns:
        # Minimal placeholder — the rebalance engine will surface missing
        # prices as DataIngestionError with a clear message.
        collated = pd.DataFrame(
            {"PLACEHOLDER": [0.0]},
            index=pd.Index(["2000-01-01"], name="Date"),
        )
        collated.to_csv(export_dir / f"{portfolio_name}_collated.csv")
        return

    # build_rebalance_plan reads collated as a wide date × ticker frame
    # with the index as dates. We create a single-row frame.
    row_data = {row["ticker"]: [row["latest_price"]] for _, row in latest.iterrows()}
    collated = pd.DataFrame(row_data, index=pd.Index(["2024-01-01"], name="Date"))
    collated.to_csv(export_dir / f"{portfolio_name}_collated.csv")
