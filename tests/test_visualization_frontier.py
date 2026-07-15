"""Tests for the efficient frontier visualization module.

.. note::

    **Canadian ETF Context** — Efficient frontier plots display CAD-
    denominated portfolio risk/return profiles.  Benchmarks use Canadian
    asset-allocation ETFs (VGRO.TO, VEQT.TO).  Frontier generation uses
    synthetic price data with fixed seeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysharpe.optimization.models import (
    OptimisationPerformance,
    OptimisationResult,
    PortfolioWeights,
)
from pysharpe.visualization.frontier import (
    generate_efficient_frontier,
    plot_portfolio_comparison,
)

# Skip tests if matplotlib is not installed
pytest.importorskip("matplotlib")


@pytest.fixture
def mock_prices() -> pd.DataFrame:
    """Provide a minimal valid price history."""
    dates = pd.date_range("2023-01-01", periods=50)
    np.random.seed(42)
    return pd.DataFrame(
        {
            "A": np.linspace(100, 110, 50) + np.random.normal(0, 1, 50),
            "B": np.linspace(50, 60, 50) + np.random.normal(0, 2, 50),
            "C": np.linspace(200, 190, 50) + np.random.normal(0, 1.5, 50),
        },
        index=dates,
    )


@pytest.fixture
def mock_portfolios() -> tuple[OptimisationResult, OptimisationResult]:
    """Provide mock user and optimized portfolios."""
    user_perf = OptimisationPerformance(
        expected_return=0.08,
        volatility=0.12,
        sharpe_ratio=0.5,
        start_date="2023-01-01",
        end_date="2023-12-31",
    )
    user_port = OptimisationResult(
        name="User",
        weights=PortfolioWeights({"A": 0.5, "B": 0.5}),
        performance=user_perf,
    )

    opt_perf = OptimisationPerformance(
        expected_return=0.15,
        volatility=0.18,
        sharpe_ratio=0.8,
        start_date="2023-01-01",
        end_date="2023-12-31",
    )
    opt_port = OptimisationResult(
        name="Opt",
        weights=PortfolioWeights({"A": 0.2, "B": 0.8}),
        performance=opt_perf,
    )

    return user_port, opt_port


@pytest.fixture
def mock_benchmarks() -> pd.DataFrame:
    """Provide mock benchmarks data."""
    return pd.DataFrame(
        {
            "Ticker": ["VGRO.TO", "VEQT.TO"],
            "Annualized Return": [0.07, 0.09],
            "Annualized Volatility": [0.10, 0.14],
            "Sharpe Ratio": [0.7, 0.64],
        }
    )


def test_generate_efficient_frontier(mock_prices: pd.DataFrame):
    """Test generating points for the efficient frontier."""
    returns, vols = generate_efficient_frontier(mock_prices, points=10)

    assert isinstance(returns, np.ndarray)
    assert isinstance(vols, np.ndarray)
    assert len(returns) == len(vols)
    assert len(returns) > 0


def test_generate_efficient_frontier_empty():
    """Test generating efficient frontier with empty dataframe."""
    with pytest.raises(ValueError, match="Prices DataFrame cannot be empty."):
        generate_efficient_frontier(pd.DataFrame())


def test_plot_portfolio_comparison(
    mock_portfolios: tuple[OptimisationResult, OptimisationResult],
    mock_benchmarks: pd.DataFrame,
    mock_prices: pd.DataFrame,
):
    """Test plotting the efficient frontier comparison."""
    import matplotlib.pyplot as plt

    user_port, opt_port = mock_portfolios

    frontier_returns = np.array([0.05, 0.10, 0.15, 0.20])
    frontier_vols = np.array([0.08, 0.12, 0.18, 0.25])

    fig = plot_portfolio_comparison(
        frontier_returns=frontier_returns,
        frontier_vols=frontier_vols,
        user_portfolio=user_port,
        optimized_portfolio=opt_port,
        benchmarks_df=mock_benchmarks,
        prices=mock_prices,
    )

    assert isinstance(fig, plt.Figure)
    axes = fig.get_axes()
    assert len(axes) == 1

    ax = axes[0]
    assert ax.get_title() == "Efficient Frontier Comparison"
    assert ax.get_xlabel() == "Annualized Volatility (Risk)"
    assert ax.get_ylabel() == "Annualized Expected Return"

    # We should have lines (frontier) and PathCollections (scatters)
    assert len(ax.lines) > 0
    assert len(ax.collections) > 0
