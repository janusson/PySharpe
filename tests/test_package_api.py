"""Tests covering the top-level pysharpe package interface.

.. note::

    **Canadian ETF Context** — The public API exposes CAD-ETF portfolio
    functions: optimisation, metrics, allocation, and rebalancing.  Lazy
    imports prevent heavy dependencies (PyMC, statsmodels) from loading
    at startup.
"""

from __future__ import annotations

import importlib
from importlib.metadata import version

import pytest

import pysharpe


def test_package_version_matches_installed_metadata():
    assert pysharpe.__version__ == version("pysharpe")
    assert "__version__" in pysharpe.__all__
    assert "__version__" in dir(pysharpe)


def test_lazy_import_exposes_metrics():
    importlib.reload(pysharpe)
    assert "sharpe_ratio" not in pysharpe.__dict__

    func = pysharpe.sharpe_ratio
    assert callable(func)
    # Cached on second access
    assert pysharpe.sharpe_ratio is func


def test_dir_lists_lazy_exports():
    names = dir(pysharpe)
    assert "optimise_portfolio" in names
    assert "compute_returns" in names


def test_return_basis_is_lazily_exported():
    basis = pysharpe.ReturnBasis.NET_OF_EXPENSES
    assert basis.value == "net_of_expenses"
    assert "ReturnBasis" in pysharpe.__all__


def test_tactical_research_validation_exports_are_lazy_and_complete():
    names = {
        "MarketDataProvenance",
        "WalkForwardConfig",
        "WalkForwardResearchResult",
        "run_walk_forward_momentum",
        "run_walk_forward_strategy",
        "TrendConfig",
        "TacticalResearchResult",
        "run_trend_strategy",
        "run_tactical_strategy",
        "PortfolioOverlapResult",
        "analyze_portfolio_overlap",
    }

    assert names.issubset(pysharpe.__all__)
    assert names.issubset(dir(pysharpe))
    assert all(getattr(pysharpe, name) is not None for name in names)


def test_unknown_attribute_raises_helpful_error():
    with pytest.raises(AttributeError) as exc:
        _ = pysharpe.does_not_exist  # noqa: F841
    assert "Available exports" in str(exc.value)
