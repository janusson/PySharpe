"""Tests for high-level workflows."""

from __future__ import annotations

import pandas as pd

from pysharpe import workflows
from pysharpe.optimization.models import OptimisationPerformance, OptimisationResult, PortfolioWeights


def test_download_portfolios_delegates(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class _StubWorkflow:
        def __init__(self, **kwargs) -> None:
            captured["init"] = kwargs

        def process_portfolios(self, **kwargs):  # noqa: D401 - signature mirrors workflow
            captured["process"] = kwargs
            return {"demo": pd.DataFrame({"AAA": [1.0]})}

    monkeypatch.setattr(workflows, "PortfolioDownloadWorkflow", _StubWorkflow)

    portfolio_dir = tmp_path / "portfolio"
    price_dir = tmp_path / "price_hist"
    export_dir = tmp_path / "exports"

    result = workflows.download_portfolios(
        portfolio_dir=portfolio_dir,
        price_history_dir=price_dir,
        export_dir=export_dir,
        period="1y",
        interval="1d",
        start="2020-01-01",
        end="2021-01-01",
    )

    assert set(result.keys()) == {"demo"}
    assert captured["init"]["portfolio_dir"] == portfolio_dir
    assert captured["process"]["period"] == "1y"
    assert captured["process"]["start"] == "2020-01-01"


def test_optimise_portfolios_skips_failed_runs(monkeypatch, tmp_path):
    collated_dir = tmp_path / "exports"
    collated_dir.mkdir()
    (collated_dir / "alpha_collated.csv").write_text("Date,AAA\n", encoding="utf-8")

    def _raise_missing(*_args, **_kwargs):
        raise FileNotFoundError("missing")

    def _should_not_be_called(**_kwargs):
        raise AssertionError("optimise_all_portfolios should not be invoked")

    monkeypatch.setattr(workflows, "optimise_portfolio", _raise_missing)
    monkeypatch.setattr(workflows, "optimise_all_portfolios", _should_not_be_called)

    result = workflows.optimise_portfolios(
        collated_dir=collated_dir,
        output_dir=collated_dir,
        make_plot=False,
    )

    assert result == {}


def test_optimise_portfolios_with_names(monkeypatch, tmp_path):
    collated_dir = tmp_path / "exports"
    collated_dir.mkdir()

    def _fake_optimize(name: str, **kwargs):  # noqa: D401
        return OptimisationResult(
            name=name,
            weights=PortfolioWeights({"AAA": 0.5, "BBB": 0.5}),
            performance=OptimisationPerformance(0.08, 0.15, 1.3),
        )

    monkeypatch.setattr(workflows, "optimise_portfolio", _fake_optimize)

    result = workflows.optimise_portfolios(
        portfolio_names=["demo"],
        collated_dir=collated_dir,
        output_dir=collated_dir,
        make_plot=False,
    )

    assert set(result.keys()) == {"demo"}
    assert isinstance(result["demo"], OptimisationResult)
