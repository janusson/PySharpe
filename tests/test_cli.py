"""Smoke tests for the CLI entrypoints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pysharpe import cli
from pysharpe.optimization.models import (
    OptimisationPerformance,
    OptimisationResult,
    PortfolioWeights,
)


def test_cli_download_smoke(monkeypatch, tmp_path, capsys):
    portfolio_dir = tmp_path / "portfolio"
    price_dir = tmp_path / "price_hist"
    export_dir_path = tmp_path / "exports"
    for directory in (portfolio_dir, price_dir, export_dir_path):
        directory.mkdir()

    def fake_download_portfolios(
        *,
        portfolio_names,
        portfolio_dir: Path,
        price_history_dir: Path,
        export_dir: Path,
        period: str,
        interval: str,
        start,
        end,
    ) -> dict[str, pd.DataFrame]:
        assert portfolio_names is None
        assert portfolio_dir == tmp_path.joinpath("portfolio").resolve()
        assert price_history_dir == price_dir.resolve()
        assert export_dir == export_dir_path.resolve()
        assert period == "max"
        assert interval == "1d"
        assert start is None and end is None
        return {"demo": pd.DataFrame({"AAA": [1.0]})}

    monkeypatch.setattr(cli.workflows, "download_portfolios", fake_download_portfolios)

    exit_code = cli.main(
        [
            "download",
            "--portfolio-dir",
            str(portfolio_dir),
            "--price-dir",
            str(price_dir),
            "--export-dir",
            str(export_dir_path),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Collated prices exported for: demo" in captured.out


def test_cli_optimize_smoke(monkeypatch, tmp_path, capsys):
    collated_root = tmp_path / "collated"
    output_root = tmp_path / "outputs"
    collated_root.mkdir()
    output_root.mkdir()

    def fake_optimise_portfolios(
        *,
        portfolio_names,
        collated_dir: Path,
        output_dir: Path,
        time_constraint: str | None,
        make_plot: bool,
    ) -> dict[str, OptimisationResult]:
        assert portfolio_names is None
        assert collated_dir == collated_root.resolve()
        assert output_dir == output_root.resolve()
        assert time_constraint is None
        assert make_plot is True
        result = OptimisationResult(
            name="demo",
            weights=PortfolioWeights({"AAA": 0.6, "BBB": 0.0}),
            performance=OptimisationPerformance(0.1, 0.2, 1.5),
        )
        return {"demo": result}

    monkeypatch.setattr(cli.workflows, "optimise_portfolios", fake_optimise_portfolios)

    exit_code = cli.main(
        [
            "optimize",
            "--collated-dir",
            str(collated_root),
            "--output-dir",
            str(output_root),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Optimised demo" in captured.out
    assert "Weights: AAA=60.00%" in captured.out
