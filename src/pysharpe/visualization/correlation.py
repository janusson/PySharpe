"""Correlation visualization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from . import utils as viz_utils

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.pyplot as plt


def plot_correlation_heatmap(
    prices: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    show: bool = False,
    title: str | None = None,
) -> plt.Axes:
    """Generate a correlation heatmap from a price DataFrame.

    Calculates daily returns and computes pairwise correlation, gracefully
    handling varying historical overlap without dropping all mismatched dates.

    Args:
        prices: DataFrame of asset prices.
        ax: Optional axes to draw onto. A new figure/axes pair is created when ``None``.
        show: When ``True`` call ``plt.show()`` before returning.
        title: Optional title for the plot.

    Returns:
        The matplotlib Axes object containing the plot.
    """
    import seaborn as sns

    plt = viz_utils.require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Calculate daily returns
    returns = prices.pct_change()

    # Calculate pairwise correlation matrix (ignores NaNs pairwise automatically)
    corr_matrix = returns.corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-0.5,
        vmax=1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )

    if title is None:
        title = "Portfolio Asset Correlation Heatmap (Daily Returns)"
    ax.set_title(title, fontweight="bold")

    fig.tight_layout()

    if show:
        plt.show()

    return ax
