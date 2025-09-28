"""Tests for the portfolio_optimization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

from pysharpe.models import PortfolioAllocation, PortfolioPerformance
from pysharpe.optimization.optimizer import OptimizationResult
from pysharpe.optimization import portfolio_optimization as p_opt


def _sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 103.0],
            "BBB": [200.0, 199.0, 201.0, 202.0],
        },
        index=dates,
    )


def test_read_price_history(tmp_path: Path):
    csv_path = tmp_path / "portfolio.csv"
    pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "AAA": [1.0, None],
            "BBB": [None, 2.0],
        }
    ).to_csv(csv_path, index=False)

    frame = p_opt.read_price_history(csv_path)

    assert list(frame.columns) == ["AAA", "BBB"]
    assert frame.loc[pd.Timestamp("2024-01-02"), "AAA"] == 1.0


def test_read_price_history_missing_file(tmp_path: Path):
    with pytest.raises(p_opt.PortfolioOptimizationError):
        p_opt.read_price_history(tmp_path / "missing.csv")


def test_window_prices_filters_and_validates():
    prices = _sample_prices()
    window = p_opt.window_prices(prices, start="2024-01-02", end="2024-01-03")
    pd.testing.assert_index_equal(
        window.index,
        pd.date_range("2024-01-02", "2024-01-03", freq="D"),
    )

    with pytest.raises(p_opt.PortfolioOptimizationError):
        p_opt.window_prices(prices, start="2030-01-01", end="2030-01-02")


def test_optimize_price_file_invokes_optimizer(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "portfolio.csv"
    prices = _sample_prices().reset_index().rename(columns={"index": "Date"})
    prices.to_csv(csv_path, index=False)

    class DummyOptimizer:
        def __init__(self, frame):
            self.frame = frame

        def max_sharpe(self, weight_bounds=(0.0, 1.0)):
            assert weight_bounds == (0.1, 0.9)
            return OptimizationResult(
                PortfolioAllocation({"AAA": 0.5, "BBB": 0.5}),
                PortfolioPerformance(0.1, 0.2, 0.5),
            )

    monkeypatch.setattr(p_opt, "PortfolioOptimizer", DummyOptimizer)

    result = p_opt.optimize_price_file(
        "test",
        csv_path,
        start="2024-01-01",
        end="2024-01-04",
        weight_bounds=(0.1, 0.9),
    )

    assert isinstance(result, OptimizationResult)


def test_batch_optimize_iterates_all(monkeypatch, tmp_path: Path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    _sample_prices().reset_index().rename(columns={"index": "Date"}).to_csv(csv_a, index=False)
    _sample_prices().reset_index().rename(columns={"index": "Date"}).to_csv(csv_b, index=False)

    seen: Dict[str, Path] = {}

    def _fake_optimize_price_file(name, path, **kwargs):  # noqa: D401 - testing stub
        seen[name] = path
        return OptimizationResult(
            PortfolioAllocation({name: 1.0}),
            PortfolioPerformance(0.1, 0.2, 0.3),
        )

    monkeypatch.setattr(p_opt, "optimize_price_file", _fake_optimize_price_file)

    result = p_opt.batch_optimize({"alpha": csv_a, "beta": csv_b})

    assert set(result.keys()) == {"alpha", "beta"}
    assert seen == {"alpha": csv_a, "beta": csv_b}


def test_save_helpers(tmp_path: Path):
    allocation = PortfolioAllocation({"AAA": 0.7, "BBB": 0.3})
    performance = PortfolioPerformance(0.1, 0.2, 0.5)
    result = OptimizationResult(allocation, performance)

    artifacts = p_opt.export_result("demo", result, tmp_path)

    assert artifacts.weights_file is not None
    assert artifacts.performance_file is not None
    assert artifacts.allocation_plot is None

    weights = pd.read_csv(artifacts.weights_file, index_col=0)
    assert set(weights.index) == {"AAA", "BBB"}

    content = artifacts.performance_file.read_text(encoding="utf-8")
    assert "Sharpe Ratio" in content


def test_optimize_portfolio_loads_collated_file(monkeypatch, tmp_path: Path):
    collated_dir = tmp_path / "exports"
    output_dir = tmp_path / "output"
    collated_dir.mkdir()

    csv_path = collated_dir / "alpha_collated.csv"
    _sample_prices().reset_index().rename(columns={"index": "Date"}).to_csv(csv_path, index=False)

    expected_result = OptimizationResult(
        PortfolioAllocation({"AAA": 0.5, "BBB": 0.5}),
        PortfolioPerformance(0.1, 0.2, 0.3),
    )

    def fake_optimize_price_file(name, path, **kwargs):  # noqa: D401 - testing stub
        assert name == "alpha"
        assert path == csv_path
        return expected_result

    exported: dict[str, Path] = {}

    def fake_export_result(name, result, destination, **kwargs):  # noqa: D401 - testing stub
        exported[name] = Path(destination)
        return p_opt.ExportArtifacts()

    monkeypatch.setattr(p_opt, "optimize_price_file", fake_optimize_price_file)
    monkeypatch.setattr(p_opt, "export_result", fake_export_result)

    outcome = p_opt.optimize_portfolio(
        "alpha",
        collated_dir=collated_dir,
        output_dir=output_dir,
        export_plot=True,
    )

    assert outcome is expected_result
    assert exported["alpha"] == output_dir


def test_optimize_all_portfolios_discovers_files(monkeypatch, tmp_path: Path):
    collated_dir = tmp_path / "exports"
    output_dir = tmp_path / "output"
    collated_dir.mkdir()

    csv_a = collated_dir / "growth_collated.csv"
    csv_b = collated_dir / "income_collated.csv"
    _sample_prices().reset_index().rename(columns={"index": "Date"}).to_csv(csv_a, index=False)
    _sample_prices().reset_index().rename(columns={"index": "Date"}).to_csv(csv_b, index=False)

    def fake_batch_optimize(price_files, **kwargs):  # noqa: D401 - testing stub
        assert set(price_files.keys()) == {"growth", "income"}
        return {
            name: OptimizationResult(
                PortfolioAllocation({name: 1.0}),
                PortfolioPerformance(0.1, 0.2, 0.3),
            )
            for name in price_files
        }

    exports: dict[str, Path] = {}

    def fake_export_result(name, result, destination, **kwargs):  # noqa: D401 - testing stub
        exports[name] = Path(destination)
        return p_opt.ExportArtifacts()

    monkeypatch.setattr(p_opt, "batch_optimize", fake_batch_optimize)
    monkeypatch.setattr(p_opt, "export_result", fake_export_result)

    results = p_opt.optimize_all_portfolios(
        collated_dir,
        output_dir=output_dir,
        export_plot=False,
    )

    assert set(results.keys()) == {"growth", "income"}
    assert exports == {"growth": output_dir, "income": output_dir}
