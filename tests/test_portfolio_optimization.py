"""Tests for the simplified portfolio optimisation module.

.. note::

    **Canadian ETF Constraints** — Optimisation is tuned for broad-market,
    CAD-denominated index ETFs.  MER values must be decimal fractions
    (< 0.10), never percentage points.  Geographic lower-bound constraints
    are dropped for regions with no mapped assets to avoid infeasible
    solver crashes.

    The efficient frontier analysis is separate from the Value Averaging
    (VA) allocation engine.  Optimisation provides analytical insight;
    the VA allocator drives actual contribution decisions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pypfopt")

import numpy as np

from pysharpe import portfolio_optimization
from pysharpe.config import ExecutionConfig
from pysharpe.optimization.models import (
    OptimisationPerformance,
    OptimisationResult,
    PortfolioWeights,
)
from pysharpe.optimization.sharpe_optimizer import SharpeOptimizerConfig
from pysharpe.portfolio_optimization import (
    ConfigurationError,
    _load_collated_prices,
    _plot_allocation,
    optimise_all_portfolios,
    optimise_from_prices,
    optimise_portfolio,
    optimise_portfolio_for_sharpe,
)


def _write_collated(tmp_path: Path, name: str) -> Path:
    frame = pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "AAA": [100.0, 101.0, 102.0, 103.0],
            "BBB": [200.0, 199.0, 201.0, 202.0],
            "CCC": [300.0, 305.0, 302.0, 301.0],
        }
    )
    csv_path = tmp_path / f"{name}_collated.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def _synth_prices(
    periods: int = 120,
    seed: int = 7,
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"),
) -> pd.DataFrame:
    """Deterministic synthetic prices with differentiated risk/return profiles."""

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=periods, freq="B")
    # (annual drift, daily volatility).  The noise is de-meaned below, so the
    # realised drift equals `annual_drift` exactly and is independent of the
    # seed — every asset keeps a reliably positive expected return above the
    # risk-free rate while still exhibiting realistic dispersion.
    profiles = {
        "AAA": (0.30, 0.010),  # high expected return, moderate vol
        "BBB": (0.12, 0.005),  # medium expected return, low vol
        "CCC": (0.05, 0.007),  # low expected return, low vol
    }
    columns: dict[str, np.ndarray] = {}
    for ticker in tickers:
        annual_drift, daily_vol = profiles.get(ticker, (0.10, 0.008))
        noise = rng.normal(0.0, 1.0, periods)
        noise -= noise.mean()
        log_returns = annual_drift / 252.0 + daily_vol * noise
        columns[ticker] = 100.0 * np.exp(np.cumsum(log_returns))
    frame = pd.DataFrame(columns, index=dates)
    frame.index.name = "Date"
    return frame


def _write_synth_collated(directory: Path, name: str, **kwargs) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{name}_collated.csv"
    _synth_prices(**kwargs).to_csv(csv_path)
    return csv_path


def _result(name: str, weights: dict[str, float] | None = None) -> OptimisationResult:
    return OptimisationResult(
        name=name,
        weights=PortfolioWeights(weights or {"AAA": 0.5, "BBB": 0.5}),
        performance=OptimisationPerformance(
            0.08, 0.15, 1.3, "2020-01-01", "2021-01-01"
        ),
    )


@pytest.fixture(autouse=True)
def _neutralise_fx(monkeypatch):
    """Keep every optimiser test hermetic and FX-free.

    ``apply_fx_conversion`` inspects each ticker via yfinance to decide whether
    to convert it into CAD.  For synthetic tickers that lookup is
    environment-dependent (it may reach the network or the local DuckDB cache),
    and a fabricated FX rate would silently distort the returns under test —
    the raw series' ~+30 %/yr became ~-47 %/yr when the FX lookup misfired.
    Replacing the module-level reference with an identity function keeps these
    tests offline and deterministic.  Tests that exercise the FX path (e.g.
    the FX-empty guard) re-patch it inside the test body, which wins.
    """

    monkeypatch.setattr(
        "pysharpe.portfolio_optimization.apply_fx_conversion",
        lambda frame, **_kwargs: frame,
    )


def test_optimise_portfolio_creates_outputs(tmp_path):
    collated_dir = tmp_path
    output_dir = tmp_path / "exports"
    output_dir.mkdir()

    _write_collated(collated_dir, "demo")

    result = portfolio_optimization.optimise_portfolio(
        "demo",
        collated_dir=collated_dir,
        output_dir=output_dir,
        make_plot=False,
        max_weight=1.0,
    )

    assert isinstance(result, OptimisationResult)
    assert set(result.weights.allocations.keys()) == {"AAA", "BBB", "CCC"}
    assert result.performance.sharpe_ratio != 0
    assert (output_dir / "demo_weights.txt").exists()
    assert (output_dir / "demo_performance.txt").exists()


def test_optimise_all_portfolios(tmp_path):
    collated_dir = tmp_path
    exports = tmp_path / "exports"
    exports.mkdir()
    _write_collated(collated_dir, "alpha")
    _write_collated(collated_dir, "beta")

    results = portfolio_optimization.optimise_all_portfolios(
        collated_dir=collated_dir,
        output_dir=exports,
        time_constraint="2023-01-02",
        max_weight=1.0,
    )

    assert {"alpha", "beta"} == set(results.keys())
    for value in results.values():
        assert isinstance(value, OptimisationResult)


def test_missing_collated_file_raises(tmp_path):
    try:
        portfolio_optimization.optimise_portfolio(
            "missing",
            collated_dir=tmp_path,
            output_dir=tmp_path,
            make_plot=False,
        )
    except FileNotFoundError:
        assert True
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected FileNotFoundError for missing collated file")


def test_optimise_portfolio_respects_constraints(tmp_path):
    collated_dir = tmp_path
    output_dir = tmp_path / "exports"
    output_dir.mkdir()

    _write_collated(collated_dir, "constrained")

    result = portfolio_optimization.optimise_portfolio(
        "constrained",
        collated_dir=collated_dir,
        output_dir=output_dir,
        asset_constraints={"max_weight": 0.65},
        make_plot=False,
        max_weight=1.0,
    )

    weights = result.weights.allocations
    assert all(weight <= 0.65 + 1e-6 for weight in weights.values())
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0


def test_optimise_portfolio_time_constraint_requires_data(tmp_path):
    collated_dir = tmp_path
    output_dir = tmp_path / "exports"
    output_dir.mkdir()
    _write_collated(collated_dir, "timing")

    with pytest.raises(ValueError):
        portfolio_optimization.optimise_portfolio(
            "timing",
            collated_dir=collated_dir,
            output_dir=output_dir,
            time_constraint="2024-01-05",
            make_plot=False,
            max_weight=1.0,
        )


def test_optimise_portfolio_skips_plot_when_disabled(monkeypatch, tmp_path):
    collated_dir = tmp_path
    output_dir = tmp_path / "exports"
    output_dir.mkdir()
    _write_collated(collated_dir, "no_plot")

    def _fail_plot(*_args, **_kwargs):
        raise AssertionError("plotting should be skipped")

    monkeypatch.setattr(portfolio_optimization, "_plot_allocation", _fail_plot)

    portfolio_optimization.optimise_portfolio(
        "no_plot",
        collated_dir=collated_dir,
        output_dir=output_dir,
        make_plot=False,
        max_weight=1.0,
    )


def test_optimise_portfolio_with_category_map(tmp_path):
    collated_dir = tmp_path
    output_dir = tmp_path / "exports"
    output_dir.mkdir()

    frame = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "SPY": [400.0, 402.0, 401.0, 405.0],
            "VOO": [400.0, 401.0, 403.0, 404.0],
            "IEF": [100.0, 100.5, 101.0, 101.5],
        }
    )
    csv_path = collated_dir / "balanced_collated.csv"
    frame.to_csv(csv_path, index=False)

    category_map = {"SPY": "US Equity", "VOO": "US Equity"}

    result = portfolio_optimization.optimise_portfolio(
        "balanced",
        collated_dir=collated_dir,
        output_dir=output_dir,
        category_map=category_map,
        make_plot=False,
        max_weight=1.0,
    )

    weights = result.weights.allocations
    assert "US Equity" in weights
    assert set(weights.keys()).issubset({"US Equity", "IEF"})


def test_optimise_portfolio_insufficient_assets(tmp_path):
    collated_dir = tmp_path
    output_dir = tmp_path / "exports"
    output_dir.mkdir()

    # Create a portfolio with only 2 assets
    frame = pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02"],
            "AAA": [100.0, 101.0],
            "BBB": [200.0, 199.0],
        }
    )
    csv_path = collated_dir / "short_collated.csv"
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError) as excinfo:
        portfolio_optimization.optimise_portfolio(
            "short",
            collated_dir=collated_dir,
            output_dir=output_dir,
            make_plot=False,
            max_weight=1.0,
        )
    assert "minimum of 3 assets" in str(excinfo.value)


def test_optimise_portfolio_for_sharpe_insufficient_assets(tmp_path):
    collated_dir = tmp_path
    output_dir = tmp_path / "exports"
    output_dir.mkdir()

    # Create a portfolio with only 2 assets
    frame = pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "AAA": [100.0, 101.0, 102.0, 103.0],
            "BBB": [200.0, 199.0, 201.0, 202.0],
        }
    )
    csv_path = collated_dir / "short_sharpe_collated.csv"
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError) as excinfo:
        portfolio_optimization.optimise_portfolio_for_sharpe(
            "short_sharpe",
            collated_dir=collated_dir,
            output_dir=output_dir,
            make_plot=False,
            max_weight=1.0,
        )
    assert "minimum of 3 assets" in str(excinfo.value)


def test_optimise_portfolio_enforces_max_weight(tmp_path):
    collated_dir = tmp_path
    output_dir = tmp_path / "exports"
    output_dir.mkdir()

    # Create a portfolio with 5 assets so max_weight=0.25 is valid.
    # Extra rows are included so that if apply_fx_conversion trims the first
    # row (no FX data before the second date), the optimiser still receives
    # enough data to converge to weights that sum to 1.0.
    frame = pd.DataFrame(
        {
            "Date": [
                "2022-12-28",
                "2022-12-29",
                "2022-12-30",
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
            ],
            "AAA": [
                95.0,
                97.0,
                99.0,
                100.0,
                105.0,
                110.0,
                120.0,
            ],  # Very strong performer
            "BBB": [198.0, 199.0, 200.0, 200.0, 199.0, 200.0, 201.0],
            "CCC": [298.0, 299.0, 300.0, 300.0, 301.0, 302.0, 303.0],
            "DDD": [405.0, 403.0, 401.0, 400.0, 395.0, 390.0, 385.0],
            "EEE": [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0],
        }
    )
    csv_path = collated_dir / "maxweight_collated.csv"
    frame.to_csv(csv_path, index=False)

    result = portfolio_optimization.optimise_portfolio(
        "maxweight",
        collated_dir=collated_dir,
        output_dir=output_dir,
        make_plot=False,
        max_weight=0.25,
    )

    weights = result.weights.allocations
    assert len(weights) > 0
    # The sum should be 1.0
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    # No individual weight should exceed 0.25
    for _ticker, weight in weights.items():
        assert weight <= 0.25 + 1e-6


def test_load_collated_prices_reflects_updated_file(tmp_path):
    """_load_collated_prices must return fresh data after the CSV is overwritten.

    In the normal optimise workflow, download_portfolios rewrites the collated CSV
    and then optimise_portfolios reads it. If the LRU cache is blind to file
    modifications, the optimizer silently uses stale prices from the prior download.
    """
    from pysharpe.portfolio_optimization import (
        _cached_collated_prices,
        _load_collated_prices,
    )

    _cached_collated_prices.cache_clear()

    csv_path = tmp_path / "demo_collated.csv"

    # Initial collated file — price is 100
    pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02"],
            "AAA": [100.0, 101.0],
            "BBB": [200.0, 201.0],
            "CCC": [300.0, 301.0],
        }
    ).to_csv(csv_path, index=False)

    result_v1 = _load_collated_prices("demo", tmp_path)
    assert result_v1["AAA"].iloc[0] == pytest.approx(100.0)

    # Simulate a re-download: overwrite the file with new prices
    pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02"],
            "AAA": [999.0, 998.0],
            "BBB": [200.0, 201.0],
            "CCC": [300.0, 301.0],
        }
    ).to_csv(csv_path, index=False)

    result_v2 = _load_collated_prices("demo", tmp_path)
    assert result_v2["AAA"].iloc[0] == pytest.approx(999.0), (
        f"Expected fresh price 999.0 but got {result_v2['AAA'].iloc[0]}. "
        "The LRU cache returned stale data from before the file was re-written."
    )


def test_optimisation_constraints(tmp_path):
    """MER constraints, geo constraints, and combined enforcement."""
    dates = pd.date_range("2020-01-01", periods=300)

    rng = np.random.default_rng(42)
    returns = pd.DataFrame(index=dates)
    returns["A"] = rng.normal(0.001, 0.02, 300)
    returns["B"] = rng.normal(0.0005, 0.005, 300)
    returns["C"] = rng.normal(0.0002, 0.01, 300)

    prices = (1 + returns).cumprod() * 100
    prices.index.name = "Date"

    collated_dir = tmp_path / "collated"
    collated_dir.mkdir()
    prices.to_csv(collated_dir / "test_portfolio_collated.csv")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mer_mapping = {"A": 0.05, "B": 0.001, "C": 0.02}
    geo_mapping = {"A": "US", "B": "CA", "C": "INT"}

    # --- MER constraint ---
    max_mer = 0.015
    result_mer = optimise_portfolio(
        "test_portfolio",
        collated_dir=collated_dir,
        output_dir=output_dir,
        mer_mapping=mer_mapping,
        max_portfolio_mer=max_mer,
        make_plot=False,
        max_weight=1.0,
    )
    weights_mer = result_mer.weights.allocations
    port_mer = sum(weights_mer.get(t, 0) * mer_mapping[t] for t in weights_mer)
    assert port_mer <= max_mer + 1e-5

    # --- Geo constraint ---
    result_geo = optimise_portfolio(
        "test_portfolio",
        collated_dir=collated_dir,
        output_dir=output_dir,
        geo_mapping=geo_mapping,
        geo_upper_bounds={"US": 0.40},
        make_plot=False,
        max_weight=1.0,
    )
    weights_geo = result_geo.weights.allocations
    assert weights_geo.get("A", 0) <= 0.40 + 1e-5

    # --- Combined constraints ---
    result_combined = optimise_portfolio(
        "test_portfolio",
        collated_dir=collated_dir,
        output_dir=output_dir,
        mer_mapping=mer_mapping,
        max_portfolio_mer=0.01,
        geo_mapping=geo_mapping,
        geo_upper_bounds={"CA": 0.6},
        make_plot=False,
        max_weight=1.0,
    )
    weights_combined = result_combined.weights.allocations
    port_mer_combined = sum(
        weights_combined.get(t, 0) * mer_mapping[t] for t in weights_combined
    )
    assert port_mer_combined <= 0.01 + 1e-5
    assert weights_combined.get("B", 0) <= 0.6 + 1e-5


def test_optimise_from_prices_converges_on_differentiated_weights():
    """With max_weight=1.0, Ledoit-Wolf shrinkage alone prevents equal-weight collapse.

    Five synthetic assets with deliberately differentiated risk/return
    profiles (high return/high vol → negative return → near-zero/low vol)
    are passed to the optimizer with no artificial per-asset ceiling.
    Covariance shrinkage must be sufficient to give the solver enough
    differentiation to allocate non-uniformly — no manual slack heuristic
    (e.g. ``1/(n-2) + 0.01``) is needed.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")

    # Five assets with deliberately differentiated returns
    returns = pd.DataFrame(index=dates)
    returns["HighRet"] = rng.normal(0.002, 0.03, 300)  # high return, high vol
    returns["MedRet"] = rng.normal(0.001, 0.02, 300)  # medium
    returns["LowRet"] = rng.normal(0.0002, 0.01, 300)  # low return, low vol
    returns["NegRet"] = rng.normal(-0.0005, 0.015, 300)  # negative return
    returns["ZeroRet"] = rng.normal(0.0, 0.005, 300)  # near-zero, very low vol

    prices = (1 + returns).cumprod() * 100

    from pysharpe.portfolio_optimization import optimise_from_prices

    result = optimise_from_prices(prices, base_currency="CAD", max_weight=1.0)

    weights = result.weights.allocations
    assert len(weights) >= 3, (
        f"Only {len(weights)} non-zero weights: {weights}. "
        "Expected at least 3 differentiated allocations."
    )

    # The weights must NOT all be equal — the optimizer must be free to
    # differentiate based on risk/return without an artificial ceiling.
    unique_weights = set(round(w, 4) for w in weights.values())
    assert len(unique_weights) > 1, (
        f"All weights are identical ({unique_weights}). "
        f"Full weights: {weights}. "
        "Equal-weight collapse means the shrinkage covariance is near-singular "
        "or the return estimates provide no differentiation."
    )

    # Risk/return differentiation must produce a meaningful weight spread
    # (well beyond numerical noise from the solver).
    weight_values = list(weights.values())
    spread = max(weight_values) - min(weight_values)
    assert spread > 0.05, (
        f"Weights are too uniform (spread={spread:.4f}): {weights}. "
        "Shrinkage covariance or return estimates are failing to "
        "differentiate assets with clearly distinct profiles."
    )


# ---------------------------------------------------------------------------
# Return-model selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["shrinkage", "ema", "constant", "mean"])
def test_optimise_from_prices_return_model_variants(model: str):
    """Every documented return model must yield a fully-invested, long-only book."""

    result = optimise_from_prices(
        _synth_prices(), return_model=model, base_currency="CAD", max_weight=1.0
    )

    weights = result.weights.allocations
    # clean_weights rounds each weight to 5 dp, so the sum may drift by ~1e-5.
    assert pytest.approx(sum(weights.values()), abs=1e-4) == 1.0
    assert all(weight >= -1e-9 for weight in weights.values())


# ---------------------------------------------------------------------------
# Canadian tax drag (TFSA / FHSA withholding tax on US-domiciled assets)
# ---------------------------------------------------------------------------


def test_optimise_from_prices_applies_us_withholding_drag_only_in_tfsa():
    """TFSA penalises US-domiciled expected returns; Non-Reg does not.

    A TFSA receives no US treaty protection, so the 15 % withholding on US
    dividends is an unrecoverable drag.  Reducing the US sleeve's expected
    return must not increase its weight, and a Non-Registered account must be
    a no-op relative to supplying no execution config at all.
    """

    prices = _synth_prices()
    proxy_map = {"AAA": {"is_us_domiciled": True}}

    baseline = optimise_from_prices(prices, base_currency="CAD", max_weight=1.0)
    tfsa = optimise_from_prices(
        prices,
        base_currency="CAD",
        max_weight=1.0,
        execution_config=ExecutionConfig(
            account_type="TFSA",
            dividend_yield_estimate=0.05,
            withholding_tax_rate=0.15,
        ),
        proxy_map=proxy_map,
    )
    non_reg = optimise_from_prices(
        prices,
        base_currency="CAD",
        max_weight=1.0,
        execution_config=ExecutionConfig(account_type="Non-Reg"),
        proxy_map=proxy_map,
    )

    baseline_aaa = baseline.weights.allocations.get("AAA", 0.0)
    tfsa_aaa = tfsa.weights.allocations.get("AAA", 0.0)

    assert pytest.approx(sum(tfsa.weights.allocations.values()), abs=1e-4) == 1.0
    # The US-domiciled sleeve is dragged down and never favoured.
    assert tfsa_aaa <= baseline_aaa + 1e-4
    assert tfsa_aaa < baseline_aaa, (
        "US withholding drag did not reduce the US-domiciled weight: "
        f"baseline={baseline_aaa:.4f}, TFSA={tfsa_aaa:.4f}"
    )
    # Non-Registered applies no drag, so it matches the baseline exactly.
    assert non_reg.weights.allocations.get("AAA", 0.0) == pytest.approx(baseline_aaa)
    # The drag itself is dividend_yield * withholding_rate (decimal fractions).
    assert ExecutionConfig(
        account_type="TFSA", dividend_yield_estimate=0.05, withholding_tax_rate=0.15
    ).annual_tax_drag == pytest.approx(0.0075)


# ---------------------------------------------------------------------------
# Geographic constraints (infeasible-lower-bound guardrail)
# ---------------------------------------------------------------------------


def test_optimise_from_prices_enforces_geo_lower_bound_for_present_region():
    """A region with mapped assets has its lower bound honoured."""

    geo = {"AAA": "US", "BBB": "CA", "CCC": "INT"}

    result = optimise_from_prices(
        _synth_prices(), geo_mapping=geo, geo_lower_bounds={"CA": 0.20}, max_weight=1.0
    )

    assert result.weights.allocations.get("BBB", 0.0) >= 0.20 - 1e-4


def test_optimise_from_prices_drops_lower_bound_for_absent_region(caplog):
    """A lower bound for an unmapped region is dropped, not enforced.

    Blindly applying it would make the problem infeasible (the Canadian
    guardrail documented in docs/GOTCHAS.md).  The optimiser must log the
    exclusion and still return a fully-invested portfolio.
    """

    geo = {"AAA": "US", "BBB": "CA", "CCC": "INT"}

    with caplog.at_level(logging.WARNING, logger="pysharpe.portfolio_optimization"):
        result = optimise_from_prices(
            _synth_prices(),
            name="geo_probe",
            geo_mapping=geo,
            geo_lower_bounds={"JP": 0.30},
            max_weight=1.0,
        )

    assert pytest.approx(sum(result.weights.allocations.values()), abs=1e-4) == 1.0
    assert any(
        "JP" in record.getMessage() and "lower bound" in record.getMessage()
        for record in caplog.records
    ), "expected a warning that the absent region's lower bound was ignored"


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_optimise_from_prices_requires_mer_for_every_ticker():
    prices = _synth_prices()

    with pytest.raises(ConfigurationError, match="MER data missing for: CCC"):
        optimise_from_prices(prices, mer_mapping={"AAA": 0.0017, "BBB": 0.0020})


def test_optimise_from_prices_requires_geo_for_every_ticker():
    prices = _synth_prices()

    with pytest.raises(ConfigurationError, match="Geographic data missing for: CCC"):
        optimise_from_prices(prices, geo_mapping={"AAA": "US", "BBB": "CA"})


# ---------------------------------------------------------------------------
# Explicit asset constraints
# ---------------------------------------------------------------------------


def test_optimise_from_prices_enforces_min_weight():
    result = optimise_from_prices(
        _synth_prices(), asset_constraints={"min_weight": 0.25}, max_weight=1.0
    )

    weights = result.weights.allocations
    # clean_weights rounds to 5 dp, so allow a small tolerance on the floor.
    assert all(weight >= 0.25 - 1e-4 for weight in weights.values())
    assert pytest.approx(sum(weights.values()), abs=1e-4) == 1.0


# ---------------------------------------------------------------------------
# Estimator failure and degenerate-metric hardening
# ---------------------------------------------------------------------------


def test_optimise_from_prices_covariance_failure_raises_runtime_error(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise ValueError("singular matrix")

    monkeypatch.setattr("pysharpe.portfolio_optimization.CovarianceShrinkage", _boom)

    with pytest.raises(RuntimeError, match="Ledoit-Wolf"):
        optimise_from_prices(_synth_prices())


def _patch_nan_volatility(monkeypatch, expected: float) -> None:
    """Force ``portfolio_performance`` to report a NaN volatility."""

    from pypfopt.efficient_frontier import EfficientFrontier as _RealEF

    class _NaNVolatilityEF(_RealEF):
        def portfolio_performance(self, *args, **kwargs):
            return (expected, float("nan"), float("nan"))

    monkeypatch.setattr(
        "pysharpe.portfolio_optimization.EfficientFrontier", _NaNVolatilityEF
    )


def test_optimise_from_prices_nan_volatility_positive_return_gives_infinite_sharpe(
    monkeypatch,
):
    """Zero-variance book with a positive return ⇒ infinite Sharpe, not NaN."""

    _patch_nan_volatility(monkeypatch, expected=0.10)

    result = optimise_from_prices(_synth_prices())

    assert result.performance.volatility == 0.0
    assert result.performance.sharpe_ratio == float("inf")


def test_optimise_from_prices_nan_volatility_low_return_gives_zero_sharpe(monkeypatch):
    """Zero-variance book at/below the risk-free rate ⇒ Sharpe 0.0 (not NaN)."""

    _patch_nan_volatility(monkeypatch, expected=0.01)

    result = optimise_from_prices(_synth_prices())

    assert result.performance.volatility == 0.0
    assert result.performance.sharpe_ratio == 0.0


# ---------------------------------------------------------------------------
# Allocation plotting strategy
# ---------------------------------------------------------------------------


def test_optimise_portfolio_make_plot_writes_allocation_png(tmp_path):
    collated = tmp_path / "collated"
    output = tmp_path / "out"
    output.mkdir()
    _write_synth_collated(collated, "demo")

    result = optimise_portfolio(
        "demo", collated_dir=collated, output_dir=output, make_plot=True, max_weight=1.0
    )

    assert isinstance(result, OptimisationResult)
    png = output / "demo_allocation.png"
    assert png.exists()
    assert png.stat().st_size > 0


def test_optimise_portfolio_plot_failure_is_non_fatal(monkeypatch, tmp_path):
    """A missing/failing plotting backend must not abort the optimisation."""

    collated = tmp_path / "collated"
    output = tmp_path / "out"
    output.mkdir()
    _write_synth_collated(collated, "demo")

    def _no_matplotlib():
        raise RuntimeError("matplotlib must be installed to render charts.")

    monkeypatch.setattr(
        "pysharpe.portfolio_optimization.require_matplotlib", _no_matplotlib
    )

    result = optimise_portfolio(
        "demo", collated_dir=collated, output_dir=output, make_plot=True, max_weight=1.0
    )

    assert isinstance(result, OptimisationResult)
    assert not (output / "demo_allocation.png").exists()


def test_plot_allocation_rejects_all_zero_weights(tmp_path):
    zeroed = _result("zeroed", {"AAA": 0.0, "BBB": 0.0})

    with pytest.raises(ValueError, match="No positive weights"):
        _plot_allocation(zeroed, tmp_path)


def test_skip_plot_strategy_logs_at_debug(tmp_path, caplog):
    from pysharpe.portfolio_optimization import _SkipAllocationPlot

    with caplog.at_level(logging.DEBUG, logger="pysharpe.portfolio_optimization"):
        _SkipAllocationPlot()(_result("skipme"), tmp_path)

    assert any(
        "Allocation plot disabled" in record.getMessage() for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Collated-price loading (forward-fill only — never back-fill)
# ---------------------------------------------------------------------------


def test_load_collated_prices_forward_fills_missing_quotes(tmp_path):
    frame = _synth_prices(periods=10)
    frame.iloc[5, 1] = np.nan  # a single missing quote must be carried forward
    (tmp_path / "gap_collated.csv").parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tmp_path / "gap_collated.csv")

    loaded = _load_collated_prices("gap", tmp_path)

    assert not loaded.isnull().values.any()
    # Forward-fill (not back-fill): the repaired value equals the prior close.
    assert loaded.iloc[5, 1] == pytest.approx(frame.iloc[4, 1])


# ---------------------------------------------------------------------------
# Sharpe-optimiser entry point (2-D tax-aware optimiser)
# ---------------------------------------------------------------------------


def test_optimise_portfolio_for_sharpe_creates_outputs(tmp_path):
    collated = tmp_path / "collated"
    output = tmp_path / "out"
    output.mkdir()
    _write_synth_collated(collated, "sharpe_demo")

    result = optimise_portfolio_for_sharpe(
        "sharpe_demo",
        collated_dir=collated,
        output_dir=output,
        make_plot=False,
        max_weight=1.0,
    )

    assert isinstance(result, OptimisationResult)
    weights = result.weights.allocations
    assert pytest.approx(sum(weights.values()), abs=1e-4) == 1.0
    assert all(weight >= -1e-9 for weight in weights.values())
    assert (output / "sharpe_demo_weights.txt").exists()
    assert (output / "sharpe_demo_performance.txt").exists()


def test_optimise_portfolio_for_sharpe_tracks_portfolio_mer(tmp_path):
    """Reported portfolio MER is the weight-weighted sum of decimal-fraction MERs."""

    collated = tmp_path / "collated"
    output = tmp_path / "out"
    output.mkdir()
    _write_synth_collated(collated, "sharpe_mer")

    mer_mapping = {"AAA": 0.0017, "BBB": 0.0020, "CCC": 0.0006}
    result = optimise_portfolio_for_sharpe(
        "sharpe_mer",
        collated_dir=collated,
        output_dir=output,
        make_plot=False,
        max_weight=1.0,
        config=SharpeOptimizerConfig(max_weight=1.0),
        mer_mapping=mer_mapping,
        max_portfolio_mer=0.0050,
    )

    weights = result.weights.allocations
    expected_mer = sum(
        weight * mer_mapping.get(t, 0.0) for t, weight in weights.items()
    )
    assert result.performance.portfolio_mer == pytest.approx(expected_mer)
    # Decimal fractions, and the 0.50 % MER ceiling is respected.
    assert result.performance.portfolio_mer is not None
    assert result.performance.portfolio_mer <= 0.0050 + 1e-4
    assert result.performance.portfolio_mer < 0.10


def test_optimise_portfolio_for_sharpe_raises_when_fx_removes_all_data(
    monkeypatch, tmp_path
):
    """If FX conversion drops every row, raise rather than optimise on nothing."""

    collated = tmp_path / "collated"
    output = tmp_path / "out"
    output.mkdir()
    _write_synth_collated(collated, "fx_empty")

    monkeypatch.setattr(
        "pysharpe.portfolio_optimization.apply_fx_conversion",
        lambda frame, **_kwargs: frame.iloc[0:0],
    )

    with pytest.raises(ValueError, match="Sharpe optimization"):
        optimise_portfolio_for_sharpe(
            "fx_empty",
            collated_dir=collated,
            output_dir=output,
            make_plot=False,
            max_weight=1.0,
        )


def test_optimise_all_portfolios_routes_to_sharpe_when_configured(
    monkeypatch, tmp_path
):
    collated = tmp_path / "collated"
    output = tmp_path / "out"
    output.mkdir()
    _write_synth_collated(collated, "demo")

    calls: list[str] = []

    def _stub(name: str, **_kwargs):  # noqa: D401
        calls.append(name)
        return _result(name)

    monkeypatch.setattr(
        "pysharpe.portfolio_optimization.optimise_portfolio_for_sharpe", _stub
    )

    results = optimise_all_portfolios(
        collated_dir=collated,
        output_dir=output,
        sharpe_optimizer_config=SharpeOptimizerConfig(),
    )

    assert calls == ["demo"]
    assert set(results) == {"demo"}


# ---------------------------------------------------------------------------
# Category mapping (collapse correlated sleeves; drop the unmapped)
# ---------------------------------------------------------------------------


def test_optimise_from_prices_wraps_category_mapping_failure(monkeypatch):
    """A category mapping that empties the frame surfaces as a clear ValueError."""

    def _boom(*_args, **_kwargs):
        raise ValueError("no overlapping data")

    monkeypatch.setattr("pysharpe.portfolio_optimization.apply_category_mapping", _boom)

    with pytest.raises(ValueError, match="after applying category mapping"):
        optimise_from_prices(
            _synth_prices(), name="cat", category_map={"AAA": "Equity"}
        )


def test_optimise_from_prices_groups_categories_and_drops_unmapped(caplog):
    """Correlated sleeves collapse into one asset; unmapped tickers are dropped."""

    tickers = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG")
    category_map = {
        "AAA": "Equity",
        "BBB": "Equity",
        "CCC": "Bonds",
        "DDD": "Bonds",
        "EEE": "Real Assets",
        "FFF": "Real Assets",
        # GGG is deliberately unmapped.
    }

    with caplog.at_level(logging.INFO, logger="pysharpe.portfolio_optimization"):
        result = optimise_from_prices(
            _synth_prices(tickers=tickers),
            name="grouped",
            category_map=category_map,
            include_unmapped_categories=False,
            max_weight=1.0,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("without category assignment" in m and "GGG" in m for m in messages)
    assert any("Grouped" in m and "Equity" in m for m in messages)

    weights = result.weights.allocations
    assert set(weights).issubset({"Equity", "Bonds", "Real Assets"})
    assert pytest.approx(sum(weights.values()), abs=1e-4) == 1.0
