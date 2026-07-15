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

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pypfopt")

from pysharpe import portfolio_optimization
from pysharpe.optimization.models import OptimisationResult


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
