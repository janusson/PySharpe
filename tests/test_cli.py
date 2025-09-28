"""Integration-style tests for the PySharpe CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pysharpe import cli
from pysharpe.models import PortfolioAllocation, PortfolioPerformance
from pysharpe.optimization.optimizer import OptimizationResult


def test_cli_download_invokes_process_portfolio(monkeypatch, tmp_path):
    portfolio_dir = tmp_path / "portfolio"
    price_dir = tmp_path / "prices"
    portfolio_dir.mkdir()
    price_dir.mkdir()

    (portfolio_dir / "growth.csv").write_text("AAPL\nMSFT\n", encoding="utf-8")

    calls = []

    def fake_process_portfolio(
        portfolio_file: Path,
        *,
        price_history_dir: Path | None = None,
        start=None,
        end=None,
        interval: str = "1d",
    ):
        calls.append((portfolio_file, price_history_dir, start, end, interval))
        return {"AAPL", "MSFT"}

    monkeypatch.setattr(cli, "process_portfolio", fake_process_portfolio)

    exit_code = cli.main(
        [
            "download",
            "--portfolio-dir",
            str(portfolio_dir),
            "--price-dir",
            str(price_dir),
        ]
    )

    assert exit_code == 0
    assert calls
    assert calls[0][0] == portfolio_dir / "growth.csv"
    assert calls[0][1] == price_dir


def test_cli_optimize_collates_and_exports(monkeypatch, tmp_path):
    portfolio_dir = tmp_path / "portfolio"
    price_dir = tmp_path / "prices"
    output_dir = tmp_path / "output"
    collated_dir = tmp_path / "collated"

    portfolio_dir.mkdir()
    price_dir.mkdir()

    (portfolio_dir / "growth.csv").write_text("AAPL\nMSFT\n", encoding="utf-8")

    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    for symbol, prices in {"AAPL": [100, 101, 102], "MSFT": [200, 201, 202]}.items():
        frame = pd.DataFrame({"Date": dates, "Close": prices})
        frame.to_csv(price_dir / f"{symbol}_hist.csv", index=False)

    allocation = PortfolioAllocation({"AAPL": 0.6, "MSFT": 0.4})
    performance = PortfolioPerformance(0.12, 0.08, 1.5)

    def fake_optimize_prices(*args, **kwargs):  # noqa: D401 - testing stub
        return OptimizationResult(allocation, performance)

    monkeypatch.setattr(cli.p_opt, "optimize_prices", fake_optimize_prices)

    exit_code = cli.main(
        [
            "optimize",
            "--portfolio-dir",
            str(portfolio_dir),
            "--price-dir",
            str(price_dir),
            "--output",
            str(output_dir),
            "--collated-dir",
            str(collated_dir),
        ]
    )

    assert exit_code == 0
    assert (collated_dir / "growth.csv").exists()
    assert any(output_dir.iterdir())
