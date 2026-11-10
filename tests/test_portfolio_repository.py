"""Tests for portfolio definition loading and the on-disk repository.

.. note::

    **Synthetic data only** — Portfolio files are written to ``tmp_path`` with
    representative CAD-denominated broad-market ETF tickers (``VFV.TO``,
    ``VCN.TO``).  No network access occurs.

    ``PortfolioRepository`` is the ingestion boundary for user-supplied
    portfolio definitions, so its skip/warn behaviour determines which
    portfolios the rest of the pipeline can silently miss.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pysharpe import config
from pysharpe.data import portfolio as portfolio_module
from pysharpe.data.portfolio import (
    PortfolioDefinition,
    PortfolioRepository,
    read_tickers,
)

# ---------------------------------------------------------------------------
# PortfolioDefinition
# ---------------------------------------------------------------------------


def test_portfolio_definition_requires_tickers():
    with pytest.raises(ValueError, match="has no tickers"):
        PortfolioDefinition("empty", (), Path("empty.csv"))


def test_portfolio_definition_ticker_set_deduplicates():
    definition = PortfolioDefinition(
        "cad_growth", ("VFV.TO", "VFV.TO", "VCN.TO"), Path("cad_growth.csv")
    )

    assert definition.ticker_set == {"VFV.TO", "VCN.TO"}


# ---------------------------------------------------------------------------
# read_tickers
# ---------------------------------------------------------------------------


def test_read_tickers_blank_file_returns_empty(tmp_path):
    path = tmp_path / "blank.csv"
    path.write_text("   \n\n", encoding="utf-8")

    assert read_tickers(path) == []


def test_read_tickers_comment_only_file_returns_empty(tmp_path):
    path = tmp_path / "comments.csv"
    path.write_text("# only comments\n# here\n", encoding="utf-8")

    assert read_tickers(path) == []


def test_read_tickers_newline_delimited_preserves_order_and_dedupes(tmp_path):
    path = tmp_path / "tickers.csv"
    path.write_text(
        "# cad portfolio\nVFV.TO\n\n VCN.TO \nVFV.TO\nQQC.TO\n",
        encoding="utf-8",
    )

    assert read_tickers(path) == ["VFV.TO", "VCN.TO", "QQC.TO"]


@pytest.mark.parametrize("header", ["ticker", "symbol", "name", "Ticker"])
def test_read_tickers_structured_csv_header(tmp_path, header):
    path = tmp_path / "portfolio.csv"
    path.write_text(
        f"id,{header},weight\n1,VFV.TO,0.6\n2,VCN.TO,0.4\n",
        encoding="utf-8",
    )

    assert read_tickers(path) == ["VFV.TO", "VCN.TO"]


def test_read_tickers_headerless_csv_treats_first_column_as_data(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text("VFV.TO,0.6\nVCN.TO,0.4\n", encoding="utf-8")

    assert read_tickers(path) == ["VFV.TO", "VCN.TO"]


def test_read_tickers_skips_rows_shorter_than_ticker_column(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text("id,symbol\n1\n2,VFV.TO\n", encoding="utf-8")

    assert read_tickers(path) == ["VFV.TO"]


@pytest.mark.parametrize("header", ["ticker", "symbol", "name", "  Ticker  "])
def test_read_tickers_bare_column_header_is_not_a_ticker(tmp_path, header):
    """Regression: a one-column CSV header must not become a phantom ticker."""

    path = tmp_path / "portfolio.csv"
    path.write_text(f"{header}\nVFV.TO\nVCN.TO\n", encoding="utf-8")

    assert read_tickers(path) == ["VFV.TO", "VCN.TO"]


def test_read_tickers_bare_column_header_only_returns_empty(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text("ticker\n", encoding="utf-8")

    assert read_tickers(path) == []


def test_read_tickers_first_row_still_data_when_not_a_header(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text("VFV.TO\nVCN.TO\n", encoding="utf-8")

    assert read_tickers(path) == ["VFV.TO", "VCN.TO"]


def test_read_tickers_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_tickers(tmp_path / "missing.csv")


# ---------------------------------------------------------------------------
# PortfolioRepository
# ---------------------------------------------------------------------------


def _settings(tmp_path: Path) -> config.PySharpeSettings:
    return config.PySharpeSettings(data_dir=tmp_path)


def _write_portfolio(directory: Path, name: str, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_repository_warns_when_directory_missing(tmp_path, caplog):
    missing = tmp_path / "does-not-exist"

    with caplog.at_level(logging.WARNING):
        repo = PortfolioRepository(settings=_settings(tmp_path), directory=missing)

    assert repo.directory == missing
    assert repo.list_portfolios() == []
    assert "Portfolio directory not found" in caplog.text


def test_repository_loads_sorted_definitions(tmp_path):
    directory = tmp_path / "portfolio"
    _write_portfolio(directory, "vfv", "ticker\nVFV.TO\n")
    _write_portfolio(directory, "cad_growth", "ticker\nVFV.TO\nVCN.TO\n")

    repo = PortfolioRepository(settings=_settings(tmp_path), directory=directory)

    assert [item.name for item in repo.list_portfolios()] == ["cad_growth", "vfv"]
    definition = repo.get_portfolio("cad_growth")
    assert definition.tickers == ("VFV.TO", "VCN.TO")
    assert definition.path == directory / "cad_growth.csv"


def test_repository_skips_empty_files_with_warning(tmp_path, caplog):
    directory = tmp_path / "portfolio"
    _write_portfolio(directory, "good", "ticker\nVFV.TO\n")
    _write_portfolio(directory, "empty", "   \n")
    _write_portfolio(directory, "comments", "# nothing here\n")

    with caplog.at_level(logging.WARNING):
        repo = PortfolioRepository(settings=_settings(tmp_path), directory=directory)

    assert [item.name for item in repo.list_portfolios()] == ["good"]
    assert "Skipping empty portfolio file: empty.csv" in caplog.text
    assert "Skipping empty portfolio file: comments.csv" in caplog.text


def test_repository_skips_file_that_disappears(tmp_path, monkeypatch, caplog):
    directory = tmp_path / "portfolio"
    _write_portfolio(directory, "ghost", "ticker\nVFV.TO\n")

    def _raise_missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(portfolio_module, "read_tickers", _raise_missing)

    with caplog.at_level(logging.ERROR):
        repo = PortfolioRepository(settings=_settings(tmp_path), directory=directory)

    assert repo.list_portfolios() == []
    assert "Portfolio file missing" in caplog.text


def test_repository_refresh_drops_removed_files(tmp_path):
    directory = tmp_path / "portfolio"
    path = _write_portfolio(directory, "vfv", "ticker\nVFV.TO\n")
    repo = PortfolioRepository(settings=_settings(tmp_path), directory=directory)
    assert repo.list_portfolios()

    path.unlink()
    repo.refresh()

    assert repo.list_portfolios() == []


def test_repository_get_portfolio_by_direct_path(tmp_path):
    directory = tmp_path / "portfolio"
    directory.mkdir()
    path = _write_portfolio(tmp_path / "elsewhere", "custom", "ticker\nQQC.TO\n")

    repo = PortfolioRepository(settings=_settings(tmp_path), directory=directory)
    definition = repo.get_portfolio(str(path))

    assert definition.name == "custom"
    assert definition.tickers == ("QQC.TO",)


def test_repository_get_portfolio_rejects_empty_direct_path(tmp_path):
    directory = tmp_path / "portfolio"
    directory.mkdir()
    path = _write_portfolio(tmp_path / "elsewhere", "empty", "   \n")

    repo = PortfolioRepository(settings=_settings(tmp_path), directory=directory)

    with pytest.raises(ValueError, match="contains no tickers"):
        repo.get_portfolio(str(path))


def test_repository_get_portfolio_missing_raises(tmp_path):
    directory = tmp_path / "portfolio"
    directory.mkdir()
    repo = PortfolioRepository(settings=_settings(tmp_path), directory=directory)

    with pytest.raises(FileNotFoundError, match="Portfolio file not found"):
        repo.get_portfolio("nope")


def test_repository_get_portfolio_directory_path_is_not_a_file(tmp_path):
    directory = tmp_path / "portfolio"
    directory.mkdir()
    repo = PortfolioRepository(settings=_settings(tmp_path), directory=directory)

    with pytest.raises(FileNotFoundError):
        repo.get_portfolio(str(directory))


def test_repository_iter_definitions_all_and_filtered(tmp_path):
    directory = tmp_path / "portfolio"
    _write_portfolio(directory, "vfv", "ticker\nVFV.TO\n")
    _write_portfolio(directory, "qqc", "ticker\nQQC.TO\n")
    repo = PortfolioRepository(settings=_settings(tmp_path), directory=directory)

    assert [item.name for item in repo.iter_definitions()] == ["qqc", "vfv"]
    assert [item.name for item in repo.iter_definitions(["vfv"])] == ["vfv"]
    assert list(repo.iter_definitions([])) == []
