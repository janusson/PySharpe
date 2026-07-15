"""Default PyMC Bayesian estimation template for asset returns.

This template demonstrates the canonical PyMC model used by the PySharpe
Bayesian optimizer. Load this file via read_file when implementing or
modifying the Bayesian estimation pipeline.

The model uses a hierarchical multivariate normal with an LKJ prior on the
correlation matrix, producing posterior expected returns and covariance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm


def build_bayesian_model(
    returns: pd.DataFrame,
    samples: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Bayesian hierarchical model to estimate expected returns and covariance.

    Args:
        returns: DataFrame of daily returns, columns = assets, rows = time.
        samples: Posterior samples per chain.
        tune: Tuning/warmup steps per chain.
        chains: Number of MCMC chains.
        random_seed: Random seed for reproducibility.

    Returns:
        (expected_returns, covariance_matrix) as numpy arrays.
        expected_returns shape: (n_assets,)
        covariance_matrix shape: (n_assets, n_assets)
    """
    n_assets: int = returns.shape[1]

    with pm.Model():
        # --- Priors ---
        # Cholesky factor of the correlation matrix (LKJ prior)
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol_cov",
            n=n_assets,
            eta=2.0,  # LKJ concentration; 2.0 favors lower correlations
            sd_dist=pm.Exponential.dist(lam=1.0, shape=n_assets),
        )

        # Covariance matrix from Cholesky factor
        pm.Deterministic("cov", chol @ chol.T)  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        # Expected returns — hierarchical prior centered at zero
        mu = pm.Normal("mu", mu=0.0, sigma=0.01, shape=n_assets)

        # --- Likelihood ---
        # Multivariate normal likelihood on observed returns
        pm.MvNormal(
            "obs",
            mu=mu,
            chol=chol,
            observed=returns.values,
        )

        # --- Sampling ---
        trace = pm.sample(
            samples,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=False,
        )

    # --- Extract posteriors ---
    expected_returns: np.ndarray = trace.posterior["mu"].mean(dim=("chain", "draw")).values
    covariance: np.ndarray = trace.posterior["cov"].mean(dim=("chain", "draw")).values

    return expected_returns, covariance
