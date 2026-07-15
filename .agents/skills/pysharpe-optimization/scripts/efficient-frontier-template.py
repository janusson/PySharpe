"""Standard PyPortfolioOpt Efficient Frontier constraint pipeline.

This template demonstrates the canonical setup used by SharpeOptimizer
in PySharpe. Load this file via read_file when implementing or modifying
the optimizer to ensure consistency with the established pipeline.

Usage patterns shown:
- Expected returns (historical mean or Bayesian posterior).
- Covariance matrix (sample or shrinkage).
- Per-asset bounds (min/max weight).
- Geographic lower-bound constraints with absent-region filtering.
- MER deduction as daily drag on expected returns.
- Solver configuration (SCS/ECOS with verbosity control).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models


def build_efficient_frontier(
    prices: pd.DataFrame,
    mers: dict[str, float] | None = None,
    geo_constraints: dict[str, dict[str, float]] | None = None,
    asset_region_map: dict[str, str] | None = None,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    solver: str = "ECOS",
) -> EfficientFrontier:
    """Construct a constrained EfficientFrontier with MER and geo constraints.

    Args:
        prices: DataFrame of daily prices, columns = tickers, index = Date.
        mers: Ticker → MER decimal fraction (e.g., {"VEQT": 0.0017}).
        geo_constraints: Region → {"lower_bound": float}.
        asset_region_map: Ticker → region name.
        weight_bounds: (min_weight, max_weight) per asset.
        solver: CVXPY solver name ("ECOS" or "SCS").

    Returns:
        A configured EfficientFrontier instance ready for optimization.
    """
    # --- Expected returns and covariance ---
    mu: pd.Series = expected_returns.mean_historical_return(prices)
    S: pd.DataFrame = risk_models.sample_cov(prices)

    # --- MER drag: subtract daily MER from expected returns ---
    if mers:
        for ticker, mer in mers.items():
            if ticker in mu.index:
                mu[ticker] -= mer / 252  # MER is decimal fraction; 252 trading days

    # --- Efficient frontier ---
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds, solver=solver)

    # --- Per-asset bounds (optional tighter constraints) ---
    # ef.add_constraint(lambda w: w >= some_custom_lower)

    # --- Geographic lower-bound constraints ---
    if geo_constraints and asset_region_map:
        # Group assets by region
        region_assets: dict[str, list[int]] = {}
        tickers: list[str] = [str(t) for t in mu.index]
        for i, ticker in enumerate(tickers):
            region = asset_region_map.get(ticker)
            if region:
                region_assets.setdefault(region, []).append(i)

        for region, constraint in geo_constraints.items():
            lower = constraint.get("lower_bound", 0.0)
            if lower <= 0.0:
                continue
            indices = region_assets.get(region, [])
            # CRITICAL: drop constraint if no assets map to this region
            if not indices:
                continue
            # Sum of weights for assets in this region >= lower_bound
            n_assets = len(mu)
            region_vector = np.zeros(n_assets)
            region_vector[indices] = 1.0
            ef.add_constraint(lambda w, v=region_vector, lb=lower: v @ w >= lb)

    return ef
