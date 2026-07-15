"""Tests covering the top-level pysharpe package interface.

.. note::

    **Canadian ETF Context** — The public API exposes CAD-ETF portfolio
    functions: optimisation, metrics, allocation, and rebalancing.  Lazy
    imports prevent heavy dependencies (PyMC, statsmodels) from loading
    at startup.
"""
from __future__ import annotations

import importlib

import pytest

import pysharpe


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


def test_unknown_attribute_raises_helpful_error():
    with pytest.raises(AttributeError) as exc:
        _ = pysharpe.does_not_exist  # noqa: F841
    assert "Available exports" in str(exc.value)
