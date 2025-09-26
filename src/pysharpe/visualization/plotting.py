"""Plotting helpers for portfolio analytics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    optimized_point: Optional[pd.Series] = None,
    ax: Optional["Axes"] = None,
    *,
    show: bool = False,
) -> "Axes":
    """Plot an efficient frontier curve with an optional optimal point."""

    import matplotlib.pyplot as plt

    # Lazily import seaborn to avoid hard dependency during package import.
    import seaborn as sns

    sns.set_style("whitegrid")

    if {"volatility", "return"} - set(frontier.columns):
        raise ValueError("Frontier data must contain 'volatility' and 'return' columns.")

    axis = ax or plt.gca()
    axis.plot(frontier["volatility"], frontier["return"], label="Efficient Frontier")

    if optimized_point is not None:
        axis.scatter(
            optimized_point["volatility"],
            optimized_point["return"],
            color="crimson",
            s=80,
            label="Max Sharpe Portfolio",
        )

    axis.set_xlabel("Annualised Volatility")
    axis.set_ylabel("Annualised Return")
    axis.legend()

    if show:
        plt.show()

    return axis
