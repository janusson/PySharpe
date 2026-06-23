"""Visualisation helpers for PySharpe."""

from .correlation import plot_correlation_heatmap
from .dca import DCAProjection, plot_dca_projection, simulate_dca
from .equity_curve import (
    plot_comparative_returns,
    plot_equity_curves,
    plot_holdings_history,
)
from .frontier import generate_efficient_frontier, plot_portfolio_comparison
from .utils import require_matplotlib

__all__ = [
    "DCAProjection",
    "simulate_dca",
    "plot_dca_projection",
    "plot_comparative_returns",
    "plot_equity_curves",
    "plot_holdings_history",
    "plot_correlation_heatmap",
    "generate_efficient_frontier",
    "plot_portfolio_comparison",
    "require_matplotlib",
]
