"""Tests for the data collector workflow helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pysharpe.data import data_collector as collector


def test_process_all_portfolios_runs_full_pipeline(monkeypatch, tmp_path: Path):
    portfolio_dir = tmp_path / "data" / "portfolio"
    price_dir = tmp_path / "data" / "price_hist"
    collated_dir = tmp_path / "data" / "exports"

    portfolio_dir.mkdir(parents=True)
    price_dir.mkdir(parents=True)

    (portfolio_dir / "growth.csv").write_text("AAA\nBBB\n", encoding="utf-8")

    def fake_download(tickers, *, destination=None, **kwargs):  # noqa: D401 - simple stub
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        data = pd.DataFrame({symbol: [1.0, 2.0] for symbol in tickers}, index=dates)
        for symbol in tickers:
            frame = pd.DataFrame({"Date": dates, "Close": [1.0, 2.0]})
            frame.to_csv(destination / f"{symbol}_hist.csv", index=False)
        return data

    monkeypatch.setattr(collector, "download_portfolio_prices", fake_download)

    results = collector.process_all_portfolios(
        portfolio_dir,
        price_history_dir=price_dir,
        collated_dir=collated_dir,
    )

    assert set(results.keys()) == {"growth"}
    assert (collated_dir / "growth_collated.csv").exists()
    assert set(results["growth"].columns) == {"AAA", "BBB"}


def test_save_collated_prices_rejects_empty_frame(tmp_path: Path):
    destination = tmp_path / "exports"
    frame = pd.DataFrame()

    try:
        collector.save_collated_prices("demo", frame, destination)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for empty frame")
