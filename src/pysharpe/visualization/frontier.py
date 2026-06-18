"""Efficient Frontier visualization and mapping."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pysharpe.optimization.models import OptimisationResult
from pysharpe.visualization import utils as viz_utils

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def generate_efficient_frontier(
    prices: pd.DataFrame, points: int = 100, frequency: int = 252
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the points of the Efficient Frontier curve.

    Args:
        prices: Historical price data with tickers as columns.
        points: Number of points to generate along the frontier.
        frequency: Observation frequency (252 for daily).

    Returns:
        tuple containing (target_returns, calculated_volatilities)
    """
    if prices.empty:
        raise ValueError("Prices DataFrame cannot be empty.")

    try:
        from pypfopt.efficient_frontier import EfficientFrontier
        from pypfopt.expected_returns import mean_historical_return
        from pypfopt.risk_models import CovarianceShrinkage
    except ImportError as exc:
        raise ImportError(
            "PyPortfolioOpt is required to generate the efficient frontier."
        ) from exc

    mu = mean_historical_return(prices, frequency=frequency)
    S = CovarianceShrinkage(prices, frequency=frequency).ledoit_wolf()

    min_return = float(mu.min())
    max_return = float(mu.max())

    # Avoid generating points if min and max returns are practically the same
    if np.isclose(min_return, max_return):
        return np.array([min_return]), np.array([np.sqrt(np.diag(S)[0])])

    target_returns = np.linspace(min_return, max_return, points)
    calculated_volatilities = []
    valid_returns = []

    for target in target_returns:
        try:
            ef = EfficientFrontier(mu, S)
            ef.efficient_return(target_return=target)
            ret, vol, _ = ef.portfolio_performance()
            calculated_volatilities.append(vol)
            valid_returns.append(ret)
        except Exception as e:
            logger.debug("Could not optimize for return %.4f: %s", target, e)

    return np.array(valid_returns), np.array(calculated_volatilities)


def plot_portfolio_comparison(
    frontier_returns: np.ndarray,
    frontier_vols: np.ndarray,
    user_portfolio: OptimisationResult,
    optimized_portfolio: OptimisationResult,
    benchmarks_df: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    frequency: int = 252,
) -> plt.Figure:
    """Plot the Efficient Frontier curve overlaid with portfolios and benchmarks.

    Args:
        frontier_returns: Array of returns forming the Efficient Frontier.
        frontier_vols: Array of volatilities corresponding to the frontier returns.
        user_portfolio: The user's current portfolio result.
        optimized_portfolio: The PySharpe optimized portfolio result.
        benchmarks_df: DataFrame containing reference benchmarks metrics.
        prices: Optional historical prices DataFrame to plot individual assets.
        frequency: Observation frequency for calculating individual asset metrics.

    Returns:
        The matplotlib Figure object.
    """
    import matplotlib.ticker as mtick

    plt = viz_utils.require_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 8))

    # 1. Plot Efficient Frontier
    if len(frontier_returns) > 0 and len(frontier_vols) > 0:
        ax.plot(
            frontier_vols,
            frontier_returns,
            "k--",
            label="Efficient Frontier",
            linewidth=2,
            zorder=1,
        )

    # 2. Plot individual assets if prices provided
    if prices is not None and not prices.empty:
        try:
            from pypfopt.expected_returns import mean_historical_return

            mu = mean_historical_return(prices, frequency=frequency)
            returns = prices.pct_change().dropna()
            vols = returns.std() * np.sqrt(frequency)

            ax.scatter(
                vols,
                mu,
                marker="o",
                color="gray",
                s=50,
                alpha=0.5,
                label="Individual Assets",
                zorder=2,
            )
            for ticker in prices.columns:
                ax.annotate(
                    ticker,
                    (vols[ticker], mu[ticker]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color="dimgray",
                )
        except Exception as e:
            logger.warning("Failed to plot individual assets: %s", e)

    # 3. Plot User's Current Portfolio Mix
    user_vol = user_portfolio.performance.volatility
    user_ret = user_portfolio.performance.expected_return
    ax.scatter(
        [user_vol],
        [user_ret],
        marker="s",
        color="blue",
        s=150,
        label="Current Portfolio",
        zorder=3,
    )

    # 4. Plot PySharpe Optimized Portfolio
    opt_vol = optimized_portfolio.performance.volatility
    opt_ret = optimized_portfolio.performance.expected_return
    ax.scatter(
        [opt_vol],
        [opt_ret],
        marker="*",
        color="green",
        s=250,
        label="Optimized Portfolio",
        zorder=4,
    )

    # 5. Plot Reference Benchmarks
    if not benchmarks_df.empty:
        required_cols = {"Ticker", "Annualized Return", "Annualized Volatility"}
        if required_cols.issubset(benchmarks_df.columns):
            bench_vols = benchmarks_df["Annualized Volatility"]
            bench_rets = benchmarks_df["Annualized Return"]
            ax.scatter(
                bench_vols,
                bench_rets,
                marker="^",
                color="red",
                s=100,
                label="Benchmarks",
                zorder=3,
            )

            for _, row in benchmarks_df.iterrows():
                ax.annotate(
                    row["Ticker"],
                    (row["Annualized Volatility"], row["Annualized Return"]),
                    xytext=(5, -10),
                    textcoords="offset points",
                    fontsize=9,
                    color="darkred",
                    fontweight="bold",
                )
        else:
            logger.warning("benchmarks_df missing required columns for plotting.")

    ax.set_title("Efficient Frontier Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Annualized Volatility (Risk)", fontsize=12)
    ax.set_ylabel("Annualized Expected Return", fontsize=12)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    return fig
