"""Top-level package for the PySharpe project."""

from .data.fetch import fetch_price_history
from .models import PortfolioAllocation, PortfolioPerformance
from .optimization.optimizer import PortfolioOptimizer
from .visualization.plotting import plot_efficient_frontier

__all__ = [
    "fetch_price_history",
    "PortfolioAllocation",
    "PortfolioPerformance",
    "PortfolioOptimizer",
    "plot_efficient_frontier",
]
