"""Baseline benchmarks for portfolio comparison."""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd

from pysharpe import metrics
from pysharpe.data.fetcher import (
    DuckDBCachedPriceFetcher,
    YFinancePriceFetcher,
    apply_fx_conversion,
)
from pysharpe.optimization.expected_returns import shrinkage_expected_return
from pysharpe.optimization.tax_location import (
    AccountType,
    AssetLocationEngine,
    AssetTaxCharacteristics,
    TaxProfile,
)

logger = logging.getLogger(__name__)

# Standard Canadian all-in-one asset allocation ETFs
CANADIAN_BENCHMARKS = {
    "VEQT.TO": "Vanguard All-Equity ETF (100/0)",
    "XEQT.TO": "iShares Core Equity ETF (100/0)",
    "VGRO.TO": "Vanguard Growth ETF (80/20)",
    "XGRO.TO": "iShares Core Growth ETF (80/20)",
    "VBAL.TO": "Vanguard Balanced ETF (60/40)",
    "XBAL.TO": "iShares Core Balanced ETF (60/40)",
}

#: Annual MERs as decimal fractions (never percentage points).  VEQT.TO uses
#: the project's canonical 0.0017 (see ``portfolio_config.json`` defaults).
BENCHMARK_MERS: dict[str, float] = {
    "VEQT.TO": 0.0017,
    "XEQT.TO": 0.0020,
    "VGRO.TO": 0.0024,
    "XGRO.TO": 0.0020,
    "VBAL.TO": 0.0024,
    "XBAL.TO": 0.0020,
}


def build_benchmark_characteristics(
    ticker: str,
    *,
    mer: float | None = None,
    dividend_yield: float | None = None,
    income_frac_interest: float | None = None,
    income_frac_eligible_dividends: float | None = None,
    income_frac_foreign_income: float | None = None,
    income_frac_capital_gains: float | None = None,
) -> AssetTaxCharacteristics:
    """Construct tax characteristics for a Canadian all-in-one benchmark ETF.

    The all-in-one funds hold US-listed securities at the fund level, so the
    unrecoverable 15 % US withholding tax is modeled via
    ``is_cad_wrapped_us_equity`` (the fund-level FWT applies in every account
    type).  Distribution composition defaults follow the funds' global
    mix: foreign income dominates, with a smaller eligible-dividend slice and
    a capital-gains component; balanced funds add an interest component from
    their bond sleeve.  These are documented estimates — pass explicit values
    to override, e.g. when a fund's prospectus indicates different splits.

    Args:
        ticker: Benchmark ticker (e.g. ``"VEQT.TO"``).
        mer: Annual MER as a decimal fraction; defaults to
            :data:`BENCHMARK_MERS` (``0.0`` when unknown).
        dividend_yield: Annual distribution yield as a decimal; defaults to
            0.018 for balanced funds and 0.015 for all-equity funds.
        income_frac_interest: float
            Interest/other-income fraction of the distribution stream.
        income_frac_eligible_dividends: float
            Eligible Canadian dividend fraction.
        income_frac_foreign_income: float
            Foreign dividend/income fraction.
        income_frac_capital_gains: float
            Realized capital-gains fraction.  The four fractions must sum
            to 1.0.

    Returns:
        AssetTaxCharacteristics for the benchmark.
    """

    key = ticker.upper()
    is_balanced = key in {"VGRO.TO", "XGRO.TO", "VBAL.TO", "XBAL.TO"}

    if mer is None:
        mer = BENCHMARK_MERS.get(key, 0.0)
    if dividend_yield is None:
        dividend_yield = 0.018 if is_balanced else 0.015
    if income_frac_interest is None:
        income_frac_interest = 0.10 if is_balanced else 0.0
    if income_frac_eligible_dividends is None:
        income_frac_eligible_dividends = 0.15
    if income_frac_foreign_income is None:
        income_frac_foreign_income = 0.50 if is_balanced else 0.60
    if income_frac_capital_gains is None:
        income_frac_capital_gains = 0.25

    return AssetTaxCharacteristics(
        ticker=key,
        dividend_yield=dividend_yield,
        is_us_domiciled=False,
        is_cad_wrapped_us_equity=True,
        income_frac_interest=income_frac_interest,
        income_frac_eligible_dividends=income_frac_eligible_dividends,
        income_frac_foreign_income=income_frac_foreign_income,
        income_frac_capital_gains=income_frac_capital_gains,
        mer=mer,
    )


def fetch_benchmark_metrics(
    tickers: list[str],
    start_date: str,
    end_date: str,
    base_currency: str = "CAD",
    *,
    reference_prices: pd.DataFrame | None = None,
    tax_profile: TaxProfile | None = None,
    asset_characteristics: dict[str, AssetTaxCharacteristics] | None = None,
    account: str = AccountType.NON_REG.value,
    shrinkage_floor: float = 0.3,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """Fetch benchmark data and compute performance metrics.

    When *tax_profile* is provided the benchmarks are evaluated with the same
    estimator and drag model as the optimized portfolio: expected returns use
    Bayes-Stein shrinkage (:func:`shrinkage_expected_return`) and are then
    reduced by MER and account-specific tax drag through the
    :class:`AssetLocationEngine` (foreign withholding tax + income tax).
    Sharpe ratios are recomputed from these adjusted returns using
    *risk_free_rate*.

    Benchmarks are evaluated jointly with *reference_prices* (the portfolio
    universe) so shrinkage pulls them toward the same cross-sectional grand
    mean as the assets they are compared against.  When no reference frame is
    provided the benchmarks are shrunk in isolation — for a single benchmark
    this degenerates to the raw historical mean (the documented fallback of
    the estimator, since no cross-section exists to shrink toward).

    Without *tax_profile* the function retains the legacy raw-metric behavior
    (geometric annualized return, unadjusted Sharpe ratio) for backward
    compatibility.

    Args:
        tickers: List of benchmark tickers (e.g., ["VEQT.TO", "VGRO.TO"]).
        start_date: ISO8601 start date for the analysis period.
        end_date: ISO8601 end date for the analysis period.
        base_currency: Target currency for evaluation (default "CAD").
        reference_prices: Optional portfolio price frame evaluated jointly
            with the benchmarks for shrinkage.
        tax_profile: Investor's marginal tax profile.  When ``None`` the
            legacy raw metrics are returned.
        asset_characteristics: Optional per-ticker overrides; benchmarks
            without an entry fall back to
            :func:`build_benchmark_characteristics`.
        account: Account label for the drag calculation.  Defaults to
            Non-Registered — the assumption is made explicit in the UI.
        shrinkage_floor: Minimum Bayes-Stein shrinkage intensity.
        risk_free_rate: Annual risk-free rate used for the Sharpe ratio.

    Returns:
        DataFrame with Ticker, Annualized Return, Annualized Volatility, and
        Sharpe Ratio.
    """

    if not tickers:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
            ]
        )

    fetcher = DuckDBCachedPriceFetcher(YFinancePriceFetcher())

    price_data: dict[str, pd.Series] = {}

    for ticker in tickers:
        try:
            df = fetcher.fetch_history(
                ticker,
                period="max",
                interval="1d",
                start=start_date,
                end=end_date,
            )
            if not df.empty:
                # Ensure index is naive DatetimeIndex for consistency with pysharpe standards
                bm_idx = df.index
                if isinstance(bm_idx, pd.DatetimeIndex) and bm_idx.tz is not None:
                    df.index = bm_idx.tz_localize(None)
                price_data[ticker] = df["Close"]
        except Exception as exc:
            logger.warning("Failed to fetch benchmark %s: %s", ticker, exc)

    if not price_data:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
            ]
        )

    prices_df = pd.DataFrame(price_data)

    # Apply FX conversion if needed
    prices_df = apply_fx_conversion(
        prices_df, base_currency=base_currency, fetcher=fetcher
    )

    if tax_profile is None:
        # Legacy raw-metrics path (backward compatibility).
        return _legacy_benchmark_metrics(prices_df)

    return _harmonized_benchmark_metrics(
        prices_df,
        reference_prices=reference_prices,
        tax_profile=tax_profile,
        asset_characteristics=asset_characteristics,
        account=account,
        shrinkage_floor=shrinkage_floor,
        risk_free_rate=risk_free_rate,
    )


def _legacy_benchmark_metrics(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw geometric returns and unadjusted Sharpe ratios.

    Retained for callers that do not supply a :class:`TaxProfile`.
    """

    # Compute returns
    returns = metrics.compute_returns(prices_df)

    # Calculate metrics
    ann_return = metrics.annualize_return(returns)
    ann_vol = metrics.annualize_volatility(returns)
    sharpe = metrics.sharpe_ratio(returns)

    # Handle single ticker edge case (metrics might return float instead of Series)
    if isinstance(ann_return, float):
        results = [
            {
                "Ticker": prices_df.columns[0],
                "Annualized Return": ann_return,
                "Annualized Volatility": ann_vol,
                "Sharpe Ratio": sharpe,
            }
        ]
    else:
        ann_return_s = cast(pd.Series, ann_return)
        ann_vol_s = cast(pd.Series, ann_vol)
        sharpe_s = cast(pd.Series, sharpe)
        results = []
        for ticker in prices_df.columns:
            results.append(
                {
                    "Ticker": ticker,
                    "Annualized Return": ann_return_s.get(ticker, 0.0),
                    "Annualized Volatility": ann_vol_s.get(ticker, 0.0),
                    "Sharpe Ratio": sharpe_s.get(ticker, 0.0),
                }
            )

    return pd.DataFrame(results)


def _harmonized_benchmark_metrics(
    prices_df: pd.DataFrame,
    *,
    reference_prices: pd.DataFrame | None,
    tax_profile: TaxProfile,
    asset_characteristics: dict[str, AssetTaxCharacteristics] | None,
    account: str,
    shrinkage_floor: float,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Compute Bayes-Stein shrunk, tax- and MER-adjusted benchmark metrics.

    Benchmarks are appended to the portfolio universe before shrinkage so
    they are pulled toward the same cross-sectional grand mean; when no
    reference frame is supplied they are shrunk in isolation (single-asset
    frames fall back to the raw mean, which is the estimator's documented
    behavior).
    """

    engine = AssetLocationEngine(tax_profile)
    chars = dict(asset_characteristics or {})
    for ticker in prices_df.columns:
        if ticker not in chars:
            chars[ticker] = build_benchmark_characteristics(ticker)

    # Evaluate benchmarks alongside the portfolio universe.
    extra_cols = [
        col
        for col in prices_df.columns
        if reference_prices is None or col not in reference_prices.columns
    ]
    if reference_prices is not None and not reference_prices.empty:
        combined = pd.concat([reference_prices, prices_df[extra_cols]], axis=1)
    else:
        combined = prices_df.copy()
    combined = combined.dropna(axis=1, how="all").dropna()
    if combined.empty:
        raise ValueError(
            "No overlapping price history between the portfolio and the "
            "requested benchmarks."
        )

    mu = shrinkage_expected_return(combined, shrinkage_floor=shrinkage_floor)

    returns = combined.pct_change().dropna()
    if returns.empty:
        raise ValueError("Insufficient benchmark price history to compute metrics.")

    volatility = metrics.annualize_volatility(returns)
    volatility_s = cast(pd.Series, volatility)  # DataFrame input → Series output

    results = []
    for ticker in prices_df.columns:
        if ticker not in mu.index:
            logger.warning(
                "Benchmark %s has no overlapping price history; skipping.", ticker
            )
            continue
        pre_tax = float(mu[ticker])
        adjusted = engine.compute_tax_adjusted_return(pre_tax, chars[ticker], account)
        vol = float(volatility_s[ticker])
        sharpe = (adjusted - risk_free_rate) / vol if vol > 0 else 0.0
        results.append(
            {
                "Ticker": ticker,
                "Annualized Return": adjusted,
                "Annualized Volatility": vol,
                "Sharpe Ratio": sharpe,
            }
        )

    return pd.DataFrame(results)
