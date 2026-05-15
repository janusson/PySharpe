"""Equity curve visualization utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import utils as viz_utils

if TYPE_CHECKING:  # pragma: no cover - type checking aide
    import matplotlib.pyplot as plt
    import pandas as pd


def plot_equity_curves(
    optimized: pd.Series,
    baseline: pd.Series,
    *,
    ax: plt.Axes | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Plot an equity curve comparing an optimized strategy against a baseline.

    Args:
        optimized: Series of the portfolio value over time for the optimized strategy.
        baseline: Series of the portfolio value over time for the baseline strategy.
        ax: Optional axes to draw onto. A new figure/axes pair is created when ``None``.
        show: When ``True`` call ``plt.show()`` before returning.
        title: Optional plot title override.

    Returns:
        Matplotlib axes containing the plot.
    """
    if ax is None:
        plt = viz_utils.require_matplotlib()
        _, ax = plt.subplots(figsize=(10, 6))
    else:
        plt = viz_utils.require_matplotlib()

    # Align indexes if possible, though matplotlib handles it gracefully usually
    ax.plot(
        optimized.index,
        optimized.values,
        label="Sharpe-Optimized Portfolio",
        linewidth=2,
        color="blue",
    )

    ax.plot(
        baseline.index,
        baseline.values,
        label="Buy-and-Hold Baseline",
        linewidth=2,
        linestyle="--",
        color="orange",
    )

    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)", fontweight="bold")

    if title is None:
        title = "Equity Curve Comparison: Optimized vs. Baseline"
    ax.set_title(title, fontweight="bold")

    # Format y-axis as dollars
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))

    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return ax
