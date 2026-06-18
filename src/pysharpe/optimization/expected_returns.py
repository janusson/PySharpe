"""Expected return estimators for portfolio optimization.

Extends PyPortfolioOpt's built-in estimators with shrinkage methods
that reduce estimation error and mitigate recency bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt.expected_returns import mean_historical_return


def shrinkage_expected_return(
    prices: pd.DataFrame,
    frequency: int = 252,
    shrinkage_floor: float = 0.0,
) -> pd.Series:
    """Bayes-Stein shrinkage estimator for expected returns.

    Shrinks individual asset mean returns toward the cross-sectional grand
    mean.  The shrinkage intensity is data-driven: noisy or similar-looking
    assets are shrunk more aggressively, while assets with strong,
    precisely-estimated differences are shrunk less.

    This directly counters recency bias — an asset whose recent returns
    are extreme will be pulled back toward the pack unless the data
    strongly indicates otherwise.

    Based on Jorion (1986) "Bayes-Stein Estimation for Portfolio Analysis".

    Args:
        prices: Historical asset prices (rows = dates, columns = tickers).
        frequency: Periods per year for annualization (252 for daily).
        shrinkage_floor: Minimum shrinkage intensity (0.0 to 1.0).
            Use 0.3–0.5 to force meaningful shrinkage even with long
            histories, which mitigates recency / look-ahead bias.

    Returns:
        Annualized expected returns as a Series indexed by ticker.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> dates = pd.date_range("2020-01-01", periods=500, freq="B")
        >>> prices = pd.DataFrame({
        ...     "A": 100 * (1 + np.random.randn(500) * 0.01).cumprod(),
        ...     "B": 100 * (1 + np.random.randn(500) * 0.015).cumprod(),
        ...     "C": 100 * (1 + np.random.randn(500) * 0.02).cumprod(),
        ... }, index=dates)
        >>> mu = shrinkage_expected_return(prices)
        >>> mu.index.tolist()
        ['A', 'B', 'C']
        >>> # Shrunk estimates should be closer together than raw means
        >>> raw = mean_historical_return(prices)
        >>> abs(mu.std()) <= abs(raw.std())
        np.True_
    """
    returns = prices.pct_change().dropna()

    n_periods, n_assets = returns.shape

    if n_assets < 2:
        return mean_historical_return(prices, frequency=frequency)

    if n_periods < n_assets + 2:
        # Not enough data for reliable shrinkage — fall back to mean
        return mean_historical_return(prices, frequency=frequency)

    # Annualized sample moments
    mu_sample = returns.mean().values * frequency  # shape (n_assets,)
    cov = returns.cov().values * frequency  # shape (n_assets, n_assets)

    # Grand mean (equal-weighted portfolio)
    mu_grand = float(mu_sample.mean())

    # ---- Shrinkage intensity (Jorion 1986, simplified) ----
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return pd.Series(np.full(n_assets, mu_grand), index=returns.columns)

    # Mahalanobis distance of sample means from grand mean.
    # Large δ² → assets really are different → less shrinkage.
    deviation = mu_sample - mu_grand
    delta_sq = float(deviation @ inv_cov @ deviation)

    if delta_sq <= 0 or not np.isfinite(delta_sq):
        phi = 1.0
    else:
        phi = (n_assets + 2) / ((n_assets + 2) + n_periods * delta_sq)
        phi = max(shrinkage_floor, min(1.0, phi))

    # Shrink toward grand mean
    mu_shrunk = mu_grand + (1.0 - phi) * (mu_sample - mu_grand)

    return pd.Series(mu_shrunk, index=returns.columns)


def constant_expected_return(prices: pd.DataFrame, frequency: int = 252) -> pd.Series:
    """Assign equal expected return to every asset (the grand mean).

    This removes return estimates entirely from the optimisation,
    reducing it to pure risk minimisation.  Useful when the investor
    has no conviction about which asset will outperform and wants
    the optimizer to focus solely on diversification.

    Args:
        prices: Historical asset prices (rows = dates, columns = tickers).
        frequency: Periods per year for annualization (252 for daily).

    Returns:
        Annualized expected returns (identical for all assets).
    """
    returns = prices.pct_change().dropna()
    grand_mean = returns.mean().mean() * frequency
    return pd.Series(grand_mean, index=returns.columns)
