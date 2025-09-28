"""Tests for visualization helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from pysharpe.visualization.plotting import plot_efficient_frontier


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("seaborn")


def test_plot_efficient_frontier_requires_columns():
    with pytest.raises(ValueError):
        plot_efficient_frontier(pd.DataFrame({"volatility": [0.1]}))


def test_plot_efficient_frontier_returns_axis():
    import matplotlib.pyplot as plt

    frontier = pd.DataFrame({"volatility": [0.1, 0.2], "return": [0.05, 0.1]})
    optimized = pd.Series({"volatility": 0.15, "return": 0.08})

    ax = plot_efficient_frontier(frontier, optimized_point=optimized, show=False)

    assert ax.get_xlabel() == "Annualised Volatility"
    assert ax.get_ylabel() == "Annualised Return"

    plt.close(ax.figure)
