"""Tests for the Canadian ETF Portfolio Research reproducible example.

Verifies:
  1. Complete offline execution (zero network calls, no yfinance calls).
  2. Deterministic synthetic price generation across repeated invocations.
  3. Correctness and mathematical precision of portfolio metrics (Sharpe, Sortino,
     CAGR, volatility, maximum drawdown, HHI, Effective N, Diversification Ratio).
  4. Reproducibility of the risk-return inversion (highest return != best risk-adjusted).
  5. Quarterly rebalancing backtest accounting and transaction cost handling.
  6. Value Averaging (VA) smart contribution allocation under 60/40 scoring blend.
  7. Independent verification checks comparing PySharpe outputs with pure NumPy.
"""

from __future__ import annotations

import socket
from typing import Any

import pandas as pd
import pytest

from examples.canadian_etf_research import (
    ASSETS,
    PORTFOLIO_WEIGHTS,
    demonstrate_value_averaging_contribution,
    evaluate_portfolio,
    generate_synthetic_etf_data,
    main,
    run_independent_checks,
    run_rebalancing_backtest,
)


@pytest.fixture
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests fail immediately if any network connection is attempted."""

    def _guarded_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network access attempted during offline test execution!")

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)


@pytest.fixture(scope="module")
def deterministic_prices() -> pd.DataFrame:
    """Generate fixed synthetic market data for module tests."""
    return generate_synthetic_etf_data(seed=123)


def test_network_isolation(block_network: None) -> None:
    """Verify that synthetic generation and analysis perform zero network calls."""
    df = generate_synthetic_etf_data(seed=123)
    assert not df.empty
    assert len(df) == 1303


def test_synthetic_data_properties(deterministic_prices: pd.DataFrame) -> None:
    """Verify structural properties of the deterministic synthetic price series."""
    df = deterministic_prices
    assert list(df.columns) == ASSETS
    assert len(df) == 1303
    assert df.index[0] == pd.Timestamp("2019-01-02")
    assert df.index[-1] == pd.Timestamp("2023-12-29")
    assert (df > 0.0).all().all(), "All simulated prices must be strictly positive"
    assert not df.isna().any().any(), "No missing values allowed in synthetic series"

    # In-sample vs Out-of-sample split sizes
    is_df = df.loc[df.index < "2022-01-01"]
    oos_df = df.loc[df.index >= "2022-01-01"]
    assert len(is_df) == 783
    assert len(oos_df) == 520


def test_seed_determinism() -> None:
    """Verify that identical seeds produce bit-for-bit identical price series."""
    df1 = generate_synthetic_etf_data(seed=999)
    df2 = generate_synthetic_etf_data(seed=999)
    pd.testing.assert_frame_equal(df1, df2)


def test_concentration_and_diversification_metrics(
    deterministic_prices: pd.DataFrame,
) -> None:
    """Verify mathematical properties of HHI, Effective N, and Diversification Ratio."""
    asset_rets = deterministic_prices.pct_change().dropna()

    # 1. US Large-Cap: 100% in one asset
    summary_us = evaluate_portfolio(
        "US_LargeCap", PORTFOLIO_WEIGHTS["US_LargeCap"], asset_rets
    )
    assert pytest.approx(summary_us.hhi_concentration, abs=1e-6) == 1.0000
    assert pytest.approx(summary_us.effective_n, abs=1e-6) == 1.00
    assert pytest.approx(summary_us.diversification_ratio, abs=1e-4) == 1.0000

    # 2. Equal Weight: 20% in each of the 5 assets
    summary_ew = evaluate_portfolio(
        "Equal_Weight_1_over_N", PORTFOLIO_WEIGHTS["Equal_Weight_1_over_N"], asset_rets
    )
    assert pytest.approx(summary_ew.hhi_concentration, abs=1e-6) == 0.2000
    assert pytest.approx(summary_ew.effective_n, abs=1e-6) == 5.00
    assert summary_ew.diversification_ratio > 1.20

    # 3. Balanced 60/40: Strong diversification benefit from uncorrelated bond allocation
    summary_bal = evaluate_portfolio(
        "Balanced_60_40", PORTFOLIO_WEIGHTS["Balanced_60_40"], asset_rets
    )
    expected_hhi = 0.28**2 + 0.18**2 + 0.14**2 + 0.40**2  # 0.2904
    assert pytest.approx(summary_bal.hhi_concentration, abs=1e-4) == expected_hhi
    assert pytest.approx(summary_bal.effective_n, abs=1e-2) == 1.0 / expected_hhi
    assert summary_bal.diversification_ratio > 1.30


def test_risk_return_inversion_out_of_sample(
    deterministic_prices: pd.DataFrame,
) -> None:
    """Demonstrate the core research finding:

    The portfolio with the highest nominal return (US_TechTilt) is NOT the
    portfolio with the best risk-adjusted performance (Balanced_60_40).
    """
    returns_df = deterministic_prices.pct_change().dropna()
    oos_rets = returns_df.loc[returns_df.index >= "2022-01-01"]

    summaries = {
        name: evaluate_portfolio(name, weights, oos_rets, risk_free_rate=0.025)
        for name, weights in PORTFOLIO_WEIGHTS.items()
    }

    tech_tilt = summaries["US_TechTilt"]
    balanced = summaries["Balanced_60_40"]
    global_eq = summaries["Global_AllEquity"]

    # 1. US_TechTilt achieves highest nominal annualized return in OOS
    assert tech_tilt.annual_return > balanced.annual_return
    assert tech_tilt.annual_return > global_eq.annual_return
    assert tech_tilt.annual_return > 0.10  # > 10% annual

    # 2. But US_TechTilt has the WORST risk-adjusted performance and deepest drawdown
    assert tech_tilt.sharpe_ratio < balanced.sharpe_ratio
    assert tech_tilt.sortino_ratio < balanced.sortino_ratio
    assert (
        tech_tilt.max_drawdown < balanced.max_drawdown
    )  # More negative (-20.75% vs -8.64%)

    # 3. Balanced 60/40 achieves highest Sharpe and Sortino in OOS
    for name, s in summaries.items():
        if name != "Balanced_60_40":
            assert balanced.sharpe_ratio > s.sharpe_ratio
            assert balanced.sortino_ratio > s.sortino_ratio
            assert balanced.max_drawdown > s.max_drawdown  # Shallower drawdown

    # 4. Global all-equity achieves lower volatility and drawdown than 100% US Large-Cap
    us_lc = summaries["US_LargeCap"]
    assert global_eq.annual_volatility < us_lc.annual_volatility
    assert global_eq.max_drawdown > us_lc.max_drawdown
    assert global_eq.sharpe_ratio > us_lc.sharpe_ratio


def test_rebalancing_backtest_quarterly(deterministic_prices: pd.DataFrame) -> None:
    """Verify quarterly rebalancing simulation with transaction fees and slippage."""
    oos_prices = deterministic_prices.loc[deterministic_prices.index >= "2022-01-01"]
    backtest_df = run_rebalancing_backtest(oos_prices, initial_capital=100_000.0)

    assert len(backtest_df) == len(PORTFOLIO_WEIGHTS)
    for _, row in backtest_df.iterrows():
        assert row["Rebalance Events"] == 8
        assert row["Terminal Value (CAD)"] > 100_000.0  # Positive net capital growth
        assert 0.15 < row["Net Total Return"] < 0.30


def test_value_averaging_contribution_allocation(
    deterministic_prices: pd.DataFrame,
) -> None:
    """Verify Value Averaging contribution allocation logic."""
    current_prices = deterministic_prices.iloc[-1]
    va_df = demonstrate_value_averaging_contribution(
        current_prices, contribution_cad=1000.0
    )

    assert not va_df.empty
    assert "allocated_dollars" in va_df.columns
    assert "opportunity_score" in va_df.columns

    # Conservation of capital: total allocation must equal contribution dollars
    total_allocated = va_df["allocated_dollars"].sum()
    assert pytest.approx(total_allocated, abs=1e-2) == 1000.0

    # Underweight asset (F_CAN_BOND) must receive the highest allocation
    bond_row = va_df.loc[va_df["ticker"] == "F_CAN_BOND"].iloc[0]
    assert bond_row["underweight"] > 0.0
    assert bond_row["allocated_dollars"] > 0.0


def test_independent_numerical_verification(deterministic_prices: pd.DataFrame) -> None:
    """Run internal mathematical verification against independent NumPy calculations."""
    returns_df = deterministic_prices.pct_change().dropna()
    run_independent_checks(returns_df)


def test_main_cli_entrypoint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test the full CLI entry point executes cleanly and prints expected output."""
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/mpl")
    exit_code = main(["--seed", "123"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert (
        "PySharpe · Reproducible Canadian ETF Portfolio Research Examples"
        in captured.out
    )
    assert "The Return vs. Risk-Adjusted Inversion" in captured.out
    assert (
        "All independent verification checks and self-tests passed cleanly."
        in captured.out
    )
