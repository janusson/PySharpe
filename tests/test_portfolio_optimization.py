"""Tests for the simplified portfolio optimisation module."""

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
    )

    assert isinstance(result, OptimisationResult)
    assert set(result.weights.allocations.keys()) == {"AAA", "BBB"}
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
    )

    weights = result.weights.allocations
    assert "US Equity" in weights
    assert set(weights.keys()).issubset({"US Equity", "IEF"})
