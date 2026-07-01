#!/usr/bin/env python3
"""Standalone verification script for the 2-D Asset Location Matrix model.

Generates a synthetic 3-asset price history, configures a realistic Canadian
tax profile and account capacities, runs the :class:`SharpeOptimizer` in 2-D
mode, and prints the resulting (ticker × account) weight matrix.

Usage
-----
    uv run python scripts/test_location_matrix.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

# Ensure the package is importable when run directly
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from pysharpe.optimization.sharpe_optimizer import (  # noqa: E402
    SharpeOptimizer,
    SharpeOptimizerConfig,
)
from pysharpe.optimization.tax_location import (  # noqa: E402
    AccountType,
    AssetTaxCharacteristics,
    TaxProfile,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_location_matrix")


# ---------------------------------------------------------------------------
# Synthetic price data
# ---------------------------------------------------------------------------
def generate_price_history(
    tickers: list[str],
    n_days: int = 504,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Produce a reproducible multi-asset price DataFrame.

    Each asset follows a geometric Brownian motion with distinct drift and
    volatility parameters so the optimiser has meaningful differentiation.

    Parameters
    ----------
    tickers : list[str]
        Asset ticker symbols (e.g. ``["VFV", "QQC", "VDY"]``).
    n_days : int
        Number of trading days (default 504 ≈ 2 calendar years).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Prices indexed by ``pd.DatetimeIndex``, columns = *tickers*.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n_days, freq="B")

    # Drift / vol per asset — intentionally different so the optimiser
    # produces non-trivial allocations.
    drifts = {"VFV": 0.0004, "QQC": 0.0005, "VDY": 0.0003}
    vols = {"VFV": 0.012, "QQC": 0.016, "VDY": 0.009}

    prices = pd.DataFrame(index=dates, dtype=float)
    for ticker in tickers:
        mu = drifts.get(ticker, 0.0004)
        sigma = vols.get(ticker, 0.012)
        returns = rng.normal(loc=mu, scale=sigma, size=n_days)
        prices[ticker] = 100.0 * np.cumprod(1 + returns)

    prices.index.name = "date"
    return prices


# ---------------------------------------------------------------------------
# Asset tax characteristics
# ---------------------------------------------------------------------------
def build_characteristics() -> dict[str, AssetTaxCharacteristics]:
    """Return realistic tax characteristics for three representative ETFs.

    * **VFV** — CAD-wrapped S&P 500 (unrecoverable US FWT).
    * **QQC** — CAD-wrapped Nasdaq-100 (unrecoverable US FWT, lower yield).
    * **VDY** — Canadian high-dividend equity (eligible dividends).
    """
    return {
        "VFV": AssetTaxCharacteristics(
            ticker="VFV",
            dividend_yield=0.012,
            is_us_domiciled=False,
            is_cad_wrapped_us_equity=True,
            income_frac_interest=0.0,
            income_frac_eligible_dividends=0.0,
            income_frac_foreign_income=1.0,
            income_frac_capital_gains=0.0,
            mer=0.0009,
        ),
        "QQC": AssetTaxCharacteristics(
            ticker="QQC",
            dividend_yield=0.005,
            is_us_domiciled=False,
            is_cad_wrapped_us_equity=True,
            income_frac_interest=0.0,
            income_frac_eligible_dividends=0.0,
            income_frac_foreign_income=1.0,
            income_frac_capital_gains=0.0,
            mer=0.0020,
        ),
        "VDY": AssetTaxCharacteristics(
            ticker="VDY",
            dividend_yield=0.035,
            is_us_domiciled=False,
            is_cad_wrapped_us_equity=False,
            income_frac_interest=0.0,
            income_frac_eligible_dividends=0.8,
            income_frac_foreign_income=0.2,
            income_frac_capital_gains=0.0,
            mer=0.0022,
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    tickers = ["VFV", "QQC", "VDY"]
    prices = generate_price_history(tickers)

    # Investor in a mid-to-high marginal bracket (45 %)
    tax_profile = TaxProfile(marginal_tax_rate=0.45)

    # Account capacities as fractions of total portfolio.
    # TFSA + RRSP are intentionally tight to force the optimiser to use
    # NON_REG for part of the allocation.
    account_capacities: dict[AccountType, float] = {
        AccountType.TFSA: 0.25,
        AccountType.RRSP: 0.35,
        AccountType.NON_REG: 0.40,
    }

    config = SharpeOptimizerConfig(
        risk_free_rate=0.03,
        tax_profile=tax_profile,
        asset_characteristics=build_characteristics(),
        account_capacities=account_capacities,
        max_weight=0.50,  # Allow individual assets up to 50 %
        num_portfolios_monte_carlo=5000,
    )

    optimizer = SharpeOptimizer(prices, config)
    result = optimizer.optimize()

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  2-D Asset Location Matrix — Optimisation Result")
    print("=" * 72)
    print(f"  Expected return : {result.expected_return:.4%}")
    print(f"  Volatility      : {result.volatility:.4%}")
    print(f"  Sharpe ratio    : {result.sharpe_ratio:.4f}")
    print(f"  Risk-free rate  : {config.risk_free_rate:.2%}")
    print("-" * 72)

    weights = result.weights
    if not weights:
        print("  (No non-zero weights returned)")
        return

    # In 2-D mode the keys are (ticker, account_value) tuples.
    # Build a pivot grid from the flat dict.
    accounts_seen: list[str] = []
    tickers_seen: list[str] = []
    for key in weights:
        if isinstance(key, tuple) and len(key) == 2:
            ticker, acct = key
            if acct not in accounts_seen:
                accounts_seen.append(acct)
            if ticker not in tickers_seen:
                tickers_seen.append(ticker)

    # Cast to the 2-D dict type so .get() accepts tuple keys cleanly
    weights_2d = cast(dict[tuple[str, str], float], weights)

    header = f"{'Ticker':<8}" + "".join(f"{a:>10}" for a in accounts_seen)
    header += f"{'Total':>10}"
    print(header)
    print("-" * len(header))

    total_by_account = {a: 0.0 for a in accounts_seen}
    for ticker in tickers_seen:
        row_parts = [f"{ticker:<8}"]
        ticker_total = 0.0
        for acct in accounts_seen:
            w = weights_2d.get((ticker, acct), 0.0)
            row_parts.append(f"{w:10.4%}")
            ticker_total += w
            total_by_account[acct] += w
        row_parts.append(f"{ticker_total:10.4%}")
        print("".join(row_parts))

    # Footer — per-account totals
    footer = f"{'Total':<8}" + "".join(
        f"{total_by_account[a]:10.4%}" for a in accounts_seen
    )
    footer += f"{sum(total_by_account.values()):10.4%}"
    print("-" * len(header))
    print(footer)

    # Verify constraints
    print("\n--- Constraint verification ---")
    grand_total = sum(weights_2d.values())
    check_ok = "✓" if abs(grand_total - 1.0) < 1e-6 else "✗"
    print(f"  Grand total weight    : {grand_total:.6f}  {check_ok}")

    for acct_type, cap in account_capacities.items():
        acct_weight = sum(w for (_, a), w in weights_2d.items() if a == acct_type.value)
        ok = "✓" if acct_weight <= cap + 1e-6 else "✗"
        print(
            f"  {acct_type.value:<8} weight : {acct_weight:.6f}  (cap {cap:.2%}) {ok}"
        )

    print("\nDone.\n")


if __name__ == "__main__":
    main()
