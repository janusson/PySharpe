"""Portfolio optimization helpers built on PyPortfolioOpt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional

import pandas as pd
from ..models import PortfolioAllocation, PortfolioPerformance

if TYPE_CHECKING:  # pragma: no cover
    from pypfopt import EfficientFrontier


@dataclass
class OptimizationResult:
    """Container for a completed optimization run."""

    allocation: PortfolioAllocation
    performance: PortfolioPerformance


class PortfolioOptimizer:
    """High-level API for maximizing the Sharpe ratio of a portfolio."""

    def __init__(self, prices: pd.DataFrame):
        if prices.empty:
            raise ValueError("Price history must not be empty.")
        if prices.isnull().all(axis=None):
            raise ValueError("Price history cannot be entirely NaN.")

        self._prices = prices.sort_index()
        self._returns: Optional[pd.DataFrame] = None

    @property
    def prices(self) -> pd.DataFrame:
        """The price history that the optimizer will use."""

        return self._prices.copy()

    @property
    def returns(self) -> pd.DataFrame:
        """Cached asset returns derived from the price history."""

        if self._returns is None:
            self._returns = self._prices.pct_change().dropna(how="all")
        return self._returns.copy()

    def _build_efficient_frontier(self) -> "EfficientFrontier":
        from pypfopt import EfficientFrontier, objective_functions, risk_models
        from pypfopt import expected_returns

        mu = expected_returns.mean_historical_return(self._prices)
        sigma = risk_models.sample_cov(self._prices)
        ef = EfficientFrontier(mu, sigma)
        ef.add_objective(objective_functions.L2_reg, gamma=0.001)
        return ef

    def max_sharpe(self, weight_bounds: tuple[float, float] = (0.0, 1.0)) -> OptimizationResult:
        """Optimize the portfolio for the maximum Sharpe ratio."""

        ef = self._build_efficient_frontier()
        ef.add_constraint(lambda w: w.sum() == 1)
        ef.bounds = weight_bounds
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        allocation = PortfolioAllocation(cleaned_weights)
        perf_stats = ef.portfolio_performance(verbose=False)
        performance = PortfolioPerformance(
            expected_annual_return=perf_stats[0],
            annual_volatility=perf_stats[1],
            sharpe_ratio=perf_stats[2],
        )
        return OptimizationResult(allocation=allocation, performance=performance)

    def rebalance(
        self, symbols: Iterable[str],
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> "PortfolioOptimizer":
        """Return a new optimizer instance restricted to the provided date window."""

        windowed = self._prices.loc[start:end, list(symbols)]
        return PortfolioOptimizer(windowed)
