"""Tests for the PySharpe CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pysharpe import cli
from pysharpe.optimization.models import OptimisationPerformance, OptimisationResult, PortfolioWeights


def _write_portfolio(directory: Path, name: str, tickers: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.csv").write_text(tickers, encoding="utf-8")


def test_optimise_subcommand_invokes_workflows(monkeypatch, tmp_path, capsys):
    portfolio_dir = tmp_path / "portfolio"
    price_dir = tmp_path / "price_hist"
    export_dir = tmp_path / "exports"
    _write_portfolio(portfolio_dir, "demo", "AAA\n")

    captured: dict[str, object] = {}

    def fake_download(*, portfolio_names, **kwargs):
        captured["download"] = {
            "portfolio_names": tuple(portfolio_names),
            "kwargs": kwargs,
        }
        return {name: pd.DataFrame({"AAA": [1.0]}) for name in portfolio_names}

    def fake_optimise(*, portfolio_names, collated_dir, output_dir, time_constraint, make_plot):
        captured["optimise"] = {
            "portfolio_names": tuple(portfolio_names),
            "collated_dir": collated_dir,
            "output_dir": output_dir,
            "time_constraint": time_constraint,
            "make_plot": make_plot,
        }
        return {
            "demo": OptimisationResult(
                name="demo",
                weights=PortfolioWeights({"AAA": 1.0}),
                performance=OptimisationPerformance(0.1, 0.2, 1.2),
            )
        }

    monkeypatch.setattr(cli.workflows, "download_portfolios", fake_download)
    monkeypatch.setattr(cli.workflows, "optimise_portfolios", fake_optimise)

    exit_code = cli.main(
        [
            "optimise",
            "--portfolio-dir",
            str(portfolio_dir),
            "--price-dir",
            str(price_dir),
            "--export-dir",
            str(export_dir),
        ]
    )

    assert exit_code == 0
    assert captured["download"]["portfolio_names"] == ("demo",)
    assert captured["optimise"]["portfolio_names"] == ("demo",)
    assert Path(captured["optimise"]["collated_dir"]).resolve() == export_dir.resolve()
    output = capsys.readouterr().out
    assert "Artefacts written" in output


def test_simulate_dca_subcommand_uses_projection(monkeypatch, capsys):
    class DummyProjection:
        def __init__(self) -> None:
            self.months = [0, 1]
            self.contributions = [100.0, 200.0]
            self.balances = [100.0, 220.0]

        def final_contribution(self) -> float:
            return 200.0

        def final_balance(self) -> float:
            return 220.0

    monkeypatch.setattr(cli, "simulate_dca", lambda **kwargs: DummyProjection())

    saved: dict[str, Path] = {}

    def fake_plot(projection):  # noqa: D401 - signature defined by CLI
        class _Axes:
            def __init__(self) -> None:
                self.figure = self

            def savefig(self, path):
                saved["path"] = Path(path)

            def clf(self):
                pass

        return _Axes()

    monkeypatch.setattr(cli, "plot_dca_projection", fake_plot)

    exit_code = cli.main(["simulate-dca", "--months", "2", "--output", "plot.png"])

    assert exit_code == 0
    assert saved["path"].name == "plot.png"
    out = capsys.readouterr().out
    assert "Final balance" in out


def test_plot_subcommand_reads_csv(monkeypatch, tmp_path):
    csv_path = tmp_path / "metrics.csv"
    pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "value": [1.0, 1.2]}).to_csv(csv_path, index=False)

    saved: dict[str, Path] = {}

    class _Figure:
        def savefig(self, path):
            saved["path"] = Path(path)

        def clf(self):
            pass

    class _Axes:
        def __init__(self) -> None:
            self.figure = _Figure()

        def set_title(self, *_args, **_kwargs):
            pass

        def set_ylabel(self, *_args, **_kwargs):
            pass

        def grid(self, *_args, **_kwargs):
            pass

    def fake_plot(self, *_, **__):  # noqa: D401 - pandas attaches bound method
        return _Axes()

    monkeypatch.setattr(pd.DataFrame, "plot", fake_plot, raising=False)

    exit_code = cli.main(
        [
            "plot",
            "--input",
            str(csv_path),
            "--date-column",
            "Date",
            "--output",
            str(tmp_path / "figure.png"),
        ]
    )

    assert exit_code == 0
    assert saved["path"].name == "figure.png"
