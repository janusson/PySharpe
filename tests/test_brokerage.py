"""Tests for the brokerage CSV export adapter.

.. note::

    **Canadian TFSA Constraint** — All buy orders are generated under the
    assumption of a Canadian Tax-Free Savings Account.  Capital gains are
    tax-exempt; no tax-loss harvesting logic is applied.

    Brokerage exports support Canadian platforms: Questrade, Wealthsimple
    (fractional shares), and Interactive Brokers.  All amounts are in CAD.

    Test tickers (VFV, QQC) are CAD-denominated broad-market ETFs.
    Single-stock placeholders (AAPL, MSFT) are synthetic only.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from pysharpe.execution.brokerage import (
    Brokerage,
    BrokerageExportConfig,
    _validate_buy_orders,
    export_buy_orders,
)
from pysharpe.execution.rebalance import (
    build_rebalance_plan,
    format_rebalance_plan,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_artifacts(export_dir: Path) -> None:
    """Write the minimum artefacts needed by build_rebalance_plan."""
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "demo_weights.txt").write_text(
        "ticker,weight\nAAPL,0.60\nMSFT,0.40\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "AAPL": [100.0, 110.0],
            "MSFT": [200.0, 210.0],
        }
    ).to_csv(export_dir / "demo_collated.csv", index=False)


def _make_sample_buy_orders() -> pd.DataFrame:
    """Return a minimal buy-orders DataFrame with two tickers."""
    return pd.DataFrame(
        {
            "ticker": ["VFV", "QQC"],
            "recommended_allocation": [3000.0, 2000.0],
            "recommended_shares": [25.0, 15.0],
            "latest_price": [120.0, 133.33],
        }
    )


def _csv_to_dicts(csv_text: str) -> list[dict[str, str]]:
    """Parse a CSV string into a list of column→value dicts."""
    return list(pd.read_csv(io.StringIO(csv_text)).to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_buy_orders_missing_column_raises():
    df = pd.DataFrame({"ticker": ["VFV"], "recommended_allocation": [100.0]})
    with pytest.raises(ValueError, match="recommended_shares"):
        _validate_buy_orders(df)


def test_validate_buy_orders_all_columns_passes():
    df = _make_sample_buy_orders()
    _validate_buy_orders(df)  # Should not raise


# ---------------------------------------------------------------------------
# Questrade
# ---------------------------------------------------------------------------


def test_questrade_produces_expected_columns():
    df = _make_sample_buy_orders()
    csv_out = export_buy_orders(df, Brokerage.QUESTRADE)
    rows = _csv_to_dicts(csv_out)

    assert len(rows) == 2
    expected_cols = {
        "Symbol",
        "Action",
        "Quantity",
        "Order Type",
        "Limit Price",
        "Duration",
    }
    assert set(rows[0].keys()) == expected_cols

    assert rows[0]["Symbol"] == "VFV"
    assert rows[0]["Action"] == "BUY"
    assert rows[0]["Quantity"] == 25
    assert rows[0]["Order Type"] == "MKT"
    # Limit Price is empty string in CSV; pd.read_csv may parse as NaN
    assert pd.isna(rows[0]["Limit Price"]) or rows[0]["Limit Price"] == ""
    assert rows[0]["Duration"] == "DAY"


def test_questrade_limit_order_includes_limit_price():
    df = _make_sample_buy_orders()
    config = BrokerageExportConfig(order_type="LMT", limit_price=120.50)
    csv_out = export_buy_orders(df, Brokerage.QUESTRADE, config=config)

    # Check the raw CSV for exact string formatting
    lines = csv_out.strip().split("\n")
    assert len(lines) >= 2  # header + at least one data row
    # First data row should contain "120.50" as the limit price
    assert "120.50" in lines[1]

    rows = _csv_to_dicts(csv_out)
    assert rows[0]["Order Type"] == "LMT"
    # pd.read_csv parses "120.50" back to float 120.5
    assert rows[0]["Limit Price"] == 120.5


def test_questrade_floors_fractional_shares():
    df = pd.DataFrame(
        {
            "ticker": ["VFV"],
            "recommended_allocation": [100.0],
            "recommended_shares": [1.9],
            "latest_price": [52.63],
        }
    )
    csv_out = export_buy_orders(df, Brokerage.QUESTRADE)
    rows = _csv_to_dicts(csv_out)
    assert rows[0]["Quantity"] == 1


def test_questrade_skips_zero_share_rows():
    df = pd.DataFrame(
        {
            "ticker": ["VFV", "QQC"],
            "recommended_allocation": [100.0, 0.0],
            "recommended_shares": [1.0, 0.0],
            "latest_price": [100.0, 50.0],
        }
    )
    csv_out = export_buy_orders(df, Brokerage.QUESTRADE)
    rows = _csv_to_dicts(csv_out)
    assert len(rows) == 1
    assert rows[0]["Symbol"] == "VFV"


def test_questrade_csv_is_valid_csv():
    df = _make_sample_buy_orders()
    csv_out = export_buy_orders(df, Brokerage.QUESTRADE)
    # Verify pd.read_csv can read it back without error
    roundtrip = pd.read_csv(io.StringIO(csv_out))
    assert len(roundtrip) == 2


# ---------------------------------------------------------------------------
# Wealthsimple
# ---------------------------------------------------------------------------


def test_wealthsimple_produces_expected_columns():
    df = _make_sample_buy_orders()
    csv_out = export_buy_orders(df, Brokerage.WEALTHSIMPLE)
    rows = _csv_to_dicts(csv_out)

    assert len(rows) == 2
    expected_cols = {
        "Ticker",
        "Action",
        "Shares",
        "Estimated Price",
        "Estimated Cost (CAD)",
    }
    assert set(rows[0].keys()) == expected_cols

    assert rows[0]["Ticker"] == "VFV"
    assert rows[0]["Action"] == "BUY"
    assert rows[0]["Estimated Price"] == "$120.00"
    assert rows[0]["Estimated Cost (CAD)"] == "$3,000.00"


def test_wealthsimple_preserves_fractional_shares():
    df = pd.DataFrame(
        {
            "ticker": ["VFV"],
            "recommended_allocation": [100.0],
            "recommended_shares": [1.9],
            "latest_price": [52.63],
        }
    )
    csv_out = export_buy_orders(df, Brokerage.WEALTHSIMPLE)

    # Check the raw CSV text: Shares should appear as "1.9" (not "1")
    lines = csv_out.strip().split("\n")
    assert len(lines) >= 2
    # The Shares column value should not be an integer
    assert "1.9" in lines[1]


def test_wealthsimple_skips_zero_share_rows():
    df = pd.DataFrame(
        {
            "ticker": ["VFV", "QQC"],
            "recommended_allocation": [100.0, 0.0],
            "recommended_shares": [1.0, 0.0],
            "latest_price": [100.0, 50.0],
        }
    )
    csv_out = export_buy_orders(df, Brokerage.WEALTHSIMPLE)
    rows = _csv_to_dicts(csv_out)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Interactive Brokers
# ---------------------------------------------------------------------------


def test_ibkr_produces_expected_columns_without_account():
    df = _make_sample_buy_orders()
    csv_out = export_buy_orders(df, Brokerage.INTERACTIVE_BROKERS)
    rows = _csv_to_dicts(csv_out)

    assert len(rows) == 2
    expected_cols = {
        "Symbol",
        "Action",
        "Quantity",
        "Order Type",
        "Limit Price",
        "Time-in-Force",
    }
    assert set(rows[0].keys()) == expected_cols
    assert "Account" not in rows[0]

    assert rows[0]["Symbol"] == "VFV"
    assert rows[0]["Action"] == "BUY"
    assert rows[0]["Quantity"] == 25
    assert rows[0]["Time-in-Force"] == "DAY"


def test_ibkr_includes_account_when_provided():
    df = _make_sample_buy_orders()
    config = BrokerageExportConfig(account="U1234567")
    csv_out = export_buy_orders(df, Brokerage.INTERACTIVE_BROKERS, config=config)
    rows = _csv_to_dicts(csv_out)

    assert "Account" in rows[0]
    assert rows[0]["Account"] == "U1234567"


def test_ibkr_limit_order_includes_limit_price():
    df = _make_sample_buy_orders()
    config = BrokerageExportConfig(order_type="LMT", limit_price=130.75)
    csv_out = export_buy_orders(df, Brokerage.INTERACTIVE_BROKERS, config=config)

    # Check the raw CSV for exact string formatting
    lines = csv_out.strip().split("\n")
    assert len(lines) >= 2
    assert "130.75" in lines[1]

    rows = _csv_to_dicts(csv_out)
    assert rows[0]["Order Type"] == "LMT"
    # pd.read_csv parses "130.75" back to float 130.75
    assert rows[0]["Limit Price"] == 130.75


def test_ibkr_floors_fractional_shares():
    df = pd.DataFrame(
        {
            "ticker": ["VFV"],
            "recommended_allocation": [100.0],
            "recommended_shares": [2.7],
            "latest_price": [37.04],
        }
    )
    csv_out = export_buy_orders(df, Brokerage.INTERACTIVE_BROKERS)
    rows = _csv_to_dicts(csv_out)
    assert rows[0]["Quantity"] == 2


def test_ibkr_skips_zero_share_rows():
    df = pd.DataFrame(
        {
            "ticker": ["VFV", "QQC"],
            "recommended_allocation": [50.0, 0.0],
            "recommended_shares": [1.0, 0.0],
            "latest_price": [50.0, 25.0],
        }
    )
    csv_out = export_buy_orders(df, Brokerage.INTERACTIVE_BROKERS)
    rows = _csv_to_dicts(csv_out)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# RebalancePlan integration
# ---------------------------------------------------------------------------


def test_accepts_rebalance_plan_directly(tmp_path):
    """The adapter should accept a full RebalancePlan, not just a raw DataFrame."""
    export_dir = tmp_path / "exports"
    _write_artifacts(export_dir)

    # Use enough cash so recommended_shares floors to at least 1 (MSFT at $210)
    plan = build_rebalance_plan(
        "demo",
        new_cash=500.0,
        holdings_mapping={"AAPL": 2},
        holdings_kind="shares",
        export_dir=export_dir,
    )

    csv_out = export_buy_orders(plan, Brokerage.QUESTRADE)
    rows = _csv_to_dicts(csv_out)

    # Only MSFT should be bought (AAPL is already overweight with 2 shares)
    assert len(rows) == 1
    assert rows[0]["Symbol"] == "MSFT"


def test_rebalance_plan_only_exports_positive_buys(tmp_path):
    """Zero-allocation rows from the plan should be excluded automatically."""
    export_dir = tmp_path / "exports"
    _write_artifacts(export_dir)

    # 10 shares of AAPL at $110 = $1,100 → heavily overweight vs 60% target
    # Use enough cash so the underweight asset (MSFT) gets at least 1 whole share
    plan = build_rebalance_plan(
        "demo",
        new_cash=500.0,
        holdings_mapping={"AAPL": 10},
        holdings_kind="shares",
        export_dir=export_dir,
    )

    csv_out = export_buy_orders(plan, Brokerage.QUESTRADE)
    rows = _csv_to_dicts(csv_out)

    # AAPL is heavily overweight — only MSFT gets an allocation
    tickers_in_csv = {r["Symbol"] for r in rows}
    assert "AAPL" not in tickers_in_csv
    assert "MSFT" in tickers_in_csv


# ---------------------------------------------------------------------------
# Multi-account filtering
# ---------------------------------------------------------------------------


def test_multi_account_filters_by_account(tmp_path):
    """When the plan has multiple accounts, specifying one should filter."""
    export_dir = tmp_path / "exports"
    _write_artifacts(export_dir)

    holdings_path = tmp_path / "holdings.csv"
    pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "account": ["TFSA", "RRSP", "TFSA", "RRSP"],
            "current_value": [110.0, 220.0, 210.0, 420.0],
            "current_shares": [1, 2, 1, 2],
        }
    ).to_csv(holdings_path, index=False)

    plan = build_rebalance_plan(
        "demo",
        new_cash=1000.0,
        holdings_csv=holdings_path,
        export_dir=export_dir,
    )

    assert plan.is_multi_account

    csv_tfsa = export_buy_orders(plan, Brokerage.QUESTRADE, account="TFSA")
    rows_tfsa = _csv_to_dicts(csv_tfsa)

    csv_rrsp = export_buy_orders(plan, Brokerage.QUESTRADE, account="RRSP")
    rows_rrsp = _csv_to_dicts(csv_rrsp)

    # Both accounts should receive orders — tickers may differ based on scoring
    assert len(rows_tfsa) >= 1
    assert len(rows_rrsp) >= 1


def test_multi_account_invalid_account_raises(tmp_path):
    """Requesting a non-existent account should raise a clear error."""
    export_dir = tmp_path / "exports"
    _write_artifacts(export_dir)

    holdings_path = tmp_path / "holdings.csv"
    pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "account": ["TFSA", "TFSA"],
            "current_value": [110.0, 210.0],
            "current_shares": [1, 1],
        }
    ).to_csv(holdings_path, index=False)

    plan = build_rebalance_plan(
        "demo",
        new_cash=500.0,
        holdings_csv=holdings_path,
        export_dir=export_dir,
    )

    with pytest.raises(ValueError, match="RRSP.*not found"):
        export_buy_orders(plan, Brokerage.QUESTRADE, account="RRSP")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_buy_orders_produces_header_only_csv():
    """An empty buy-orders DataFrame should produce CSV with only the header."""
    df = pd.DataFrame(
        columns=[
            "ticker",
            "recommended_allocation",
            "recommended_shares",
            "latest_price",
        ]
    )
    csv_out = export_buy_orders(df, Brokerage.QUESTRADE)
    lines = csv_out.strip().split("\n")
    assert len(lines) == 1  # Header only
    assert "Symbol" in lines[0]


def test_default_config_is_market_day():
    """When no config is passed, market order + DAY should be used."""
    df = _make_sample_buy_orders()
    csv_out = export_buy_orders(df, Brokerage.INTERACTIVE_BROKERS)
    rows = _csv_to_dicts(csv_out)

    assert all(r["Order Type"] == "MKT" for r in rows)
    assert all(r["Time-in-Force"] == "DAY" for r in rows)


def test_brokerage_enum_values():
    """Verify the enum string values for documentation accuracy."""
    assert Brokerage.QUESTRADE.value == "questrade"
    assert Brokerage.WEALTHSIMPLE.value == "wealthsimple"
    assert Brokerage.INTERACTIVE_BROKERS.value == "interactive_brokers"


def test_export_config_limit_price_str_default():
    config = BrokerageExportConfig()
    assert config.limit_price_str == ""


def test_export_config_limit_price_str_set():
    config = BrokerageExportConfig(limit_price=99.95)
    assert config.limit_price_str == "99.95"


# ---------------------------------------------------------------------------
# DataFrame acceptance (non-RebalancePlan path)
# ---------------------------------------------------------------------------


def test_raw_dataframe_with_extra_columns_still_works():
    """A DataFrame with bonus columns (like from scored_state) should work."""
    df = pd.DataFrame(
        {
            "ticker": ["VFV"],
            "recommended_allocation": [1000.0],
            "recommended_shares": [8.0],
            "latest_price": [125.0],
            "opportunity_score": [0.85],  # Extra column — should be ignored
            "target_weight": [0.40],
        }
    )
    csv_out = export_buy_orders(df, Brokerage.QUESTRADE)
    rows = _csv_to_dicts(csv_out)
    assert len(rows) == 1
    assert rows[0]["Symbol"] == "VFV"


def test_raw_dataframe_missing_column_raises():
    df = pd.DataFrame({"ticker": ["VFV"], "recommended_allocation": [100.0]})
    with pytest.raises(ValueError, match="recommended_shares"):
        export_buy_orders(df, Brokerage.QUESTRADE)


def _write_rebalance_artifacts(export_dir) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "demo_weights.txt").write_text(
        "ticker,weight\nAAPL,0.60\nMSFT,0.40\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "AAPL": [100.0, 110.0],
            "MSFT": [200.0, 210.0],
        }
    ).to_csv(export_dir / "demo_collated.csv", index=False)


def test_build_rebalance_plan_allocates_to_underweight_asset(tmp_path):
    export_dir = tmp_path / "exports"
    _write_rebalance_artifacts(export_dir)

    plan = build_rebalance_plan(
        "demo",
        new_cash=100.0,
        holdings_mapping={"AAPL": 2},
        holdings_kind="shares",
        export_dir=export_dir,
    )

    buy_orders = plan.buy_orders

    assert list(buy_orders["ticker"]) == ["MSFT"]
    assert buy_orders.iloc[0]["recommended_allocation"] == pytest.approx(100.0)
    assert buy_orders.iloc[0]["recommended_shares"] == pytest.approx(100.0 / 210.0)
    assert "opportunity_score" in plan.scored_state.columns

    rendered = format_rebalance_plan(plan)
    assert "Rebalance plan for demo" in rendered
    assert "MSFT" in rendered
    assert "$100.00" in rendered


def test_rebalance_main_accepts_json_file(tmp_path, capsys):
    export_dir = tmp_path / "exports"
    _write_rebalance_artifacts(export_dir)

    holdings_path = tmp_path / "holdings.json"
    holdings_path.write_text(json.dumps({"AAPL": 2}), encoding="utf-8")

    exit_code = main(
        [
            "--portfolio",
            "demo",
            "--holdings-json",
            str(holdings_path),
            "--holdings-kind",
            "shares",
            "--new-cash",
            "100",
            "--export-dir",
            str(export_dir),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Rebalance plan for demo" in output
    assert "MSFT" in output
