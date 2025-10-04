"""Tests for the simplified CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pysharpe import cli
from pysharpe.optimization.models import OptimisationPerformance, OptimisationResult, PortfolioWeights


def _write_portfolio(directory: Path, name: str, tickers: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.csv").write_text(tickers, encoding="utf-8")


def test_cli_runs_full_workflow(monkeypatch, tmp_path, capsys):
    portfolio_dir = tmp_path / "portfolio"
    price_dir = tmp_path / "price_hist"
    export_dir = tmp_path / "exports"
    for directory in (price_dir, export_dir):
        directory.mkdir()
    _write_portfolio(portfolio_dir, "demo", "AAA\nBBB\n")

    captured: dict[str, object] = {}

    def fake_download_portfolios(*, portfolio_names, portfolio_dir: Path, price_history_dir: Path, export_dir: Path, period: str, interval: str, start, end):
        captured["download"] = {
            "portfolio_names": tuple(portfolio_names),
            "portfolio_dir": portfolio_dir,
            "price_history_dir": price_history_dir,
            "export_dir": export_dir,
            "period": period,
            "interval": interval,
            "start": start,
            "end": end,
        }
        return {"demo": pd.DataFrame({"AAA": [1.0]})}

    def fake_optimise_portfolios(*, portfolio_names, collated_dir: Path, output_dir: Path, time_constraint, make_plot):
        captured["optimise"] = {
            "portfolio_names": tuple(portfolio_names),
            "collated_dir": collated_dir,
            "output_dir": output_dir,
            "time_constraint": time_constraint,
            "make_plot": make_plot,
        }
        result = OptimisationResult(
            name="demo",
            weights=PortfolioWeights({"AAA": 0.5, "BBB": 0.5}),
            performance=OptimisationPerformance(0.1, 0.2, 1.5),
        )
        return {"demo": result}

    monkeypatch.setattr(cli.workflows, "download_portfolios", fake_download_portfolios)
    monkeypatch.setattr(cli.workflows, "optimise_portfolios", fake_optimise_portfolios)

    exit_code = cli.main(
        [
            "--portfolio-dir",
            str(portfolio_dir),
            "--price-dir",
            str(price_dir),
            "--export-dir",
            str(export_dir),
        ]
    )

    assert exit_code == 0
    assert tuple(captured["download"]["portfolio_names"]) == ("demo",)
    assert captured["download"]["portfolio_dir"] == portfolio_dir.resolve()
    assert captured["download"]["price_history_dir"] == price_dir.resolve()
    assert captured["download"]["export_dir"] == export_dir.resolve()
    assert tuple(captured["optimise"]["portfolio_names"]) == ("demo",)
    assert captured["download"]["period"] == cli.DEFAULT_PERIOD
    assert captured["download"]["interval"] == cli.DEFAULT_INTERVAL
    assert captured["optimise"]["collated_dir"] == export_dir.resolve()
    assert captured["optimise"]["output_dir"] == export_dir.resolve()
    assert captured["optimise"]["time_constraint"] is None
    assert captured["optimise"]["make_plot"] is True

    output = capsys.readouterr().out
    assert "Portfolio definitions directory" in output
    assert "Available portfolios" in output
    assert "Optimised demo" in output
    assert "Weights: AAA=50.00%" in output


def test_cli_skip_download(monkeypatch, tmp_path, capsys):
    portfolio_dir = tmp_path / "portfolio"
    price_dir = tmp_path / "price_hist"
    export_dir = tmp_path / "exports"
    _write_portfolio(portfolio_dir, "demo", "AAA\n")

    def _fail_download(**_kwargs):
        raise AssertionError("download_portfolios should not be invoked")

    def fake_optimise_portfolios(*, portfolio_names, collated_dir: Path, output_dir: Path, time_constraint, make_plot):
        return {
            "demo": OptimisationResult(
                name="demo",
                weights=PortfolioWeights({"AAA": 1.0}),
                performance=OptimisationPerformance(0.1, 0.2, 1.4),
            )
        }

    monkeypatch.setattr(cli.workflows, "download_portfolios", _fail_download)
    monkeypatch.setattr(cli.workflows, "optimise_portfolios", fake_optimise_portfolios)

    exit_code = cli.main(
        [
            "--portfolio-dir",
            str(portfolio_dir),
            "--price-dir",
            str(price_dir),
            "--export-dir",
            str(export_dir),
            "--skip-download",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Skipping download step" in output
    assert "Optimised demo" in output


def test_cli_reports_missing_portfolios(tmp_path, capsys):
    portfolio_dir = tmp_path / "portfolio"
    price_dir = tmp_path / "price_hist"
    export_dir = tmp_path / "exports"

    exit_code = cli.main(
        [
            "--portfolio-dir",
            str(portfolio_dir),
            "--price-dir",
            str(price_dir),
            "--export-dir",
            str(export_dir),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "No portfolio CSV files found" in output
