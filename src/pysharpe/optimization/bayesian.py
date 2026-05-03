"""Bayesian Portfolio Optimization using PyMC.

This module provides tools for estimating the posterior distribution of asset returns
using Bayesian methods, serving as a robust foundation for portfolio allocation models
like Black-Litterman.
"""

import logging
from typing import Dict, Optional, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """Estimates the posterior distribution of asset returns using PyMC."""

    def __init__(self, random_seed: Optional[int] = 42):
        """Initialize the Bayesian Optimizer.

        Args:
            random_seed (Optional[int]): Seed for reproducibility of MCMC sampling.
        """
        self.random_seed = random_seed
        self.trace_ = None
        self.model_ = None
        self.assets_ = None

    def fit_returns_model(
        self,
        returns: pd.DataFrame,
        draws: int = 1000,
        tune: int = 1000,
        target_accept: float = 0.9,
        **kwargs,
    ) -> az.InferenceData:
        """Fit a Bayesian multivariate normal model to historical returns.

        This method estimates the mean vector and covariance matrix of asset returns.

        Args:
            returns (pd.DataFrame): Historical asset returns (rows are dates, columns are assets).
            draws (int): Number of MCMC samples to draw.
            tune (int): Number of tuning steps for the sampler.
            target_accept (float): Target acceptance rate for NUTS.
            **kwargs: Additional keyword arguments to pass to pm.sample().

        Returns:
            az.InferenceData: The trace containing posterior samples.
        """
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("Returns must be a pandas DataFrame.")
        if returns.empty:
            raise ValueError("Returns DataFrame cannot be empty.")

        self.assets_ = returns.columns.tolist()
        n_assets = len(self.assets_)
        data = returns.values

        with pm.Model() as self.model_:
            # Priors for the mean returns
            mu = pm.Normal("mu", mu=0.0, sigma=0.1, shape=n_assets)

            # Priors for the covariance matrix using LKJ Cholesky
            # This is standard practice in PyMC for covariance matrices
            sd_dist = pm.HalfCauchy.dist(beta=0.1, shape=n_assets)
            chol, corr, stds = pm.LKJCholeskyCov(
                "chol", n=n_assets, eta=2.0, sd_dist=sd_dist, compute_corr=True
            )

            # Reconstruct the covariance matrix as a Deterministic node for easy extraction
            cov = pm.Deterministic("cov", pm.math.dot(chol, chol.T))

            # Likelihood
            pm.MvNormal("obs", mu=mu, chol=chol, observed=data)

            # Sample from the posterior
            logger.info(
                f"Starting MCMC sampling with {draws} draws and {tune} tuning steps..."
            )
            self.trace_ = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=target_accept,
                random_seed=self.random_seed,
                return_inferencedata=True,
                progressbar=False,  # Keep logs clean
                **kwargs,
            )

        logger.info("MCMC sampling completed.")
        return self.trace_

    def get_posterior_estimates(self) -> Tuple[pd.Series, pd.DataFrame]:
        """Extract the posterior mean returns and covariance matrix.

        Returns:
            Tuple[pd.Series, pd.DataFrame]: A tuple containing:
                - Expected returns (posterior mean of 'mu').
                - Expected covariance matrix (posterior mean of 'cov').

        Raises:
            RuntimeError: If `fit_returns_model` has not been called yet.
        """
        if self.trace_ is None or self.model_ is None:
            raise RuntimeError(
                "Model has not been fitted. Call `fit_returns_model` first."
            )

        # Extract posterior samples
        posterior = self.trace_.posterior

        # Calculate expected returns (mean of the posterior distribution for 'mu')
        expected_returns = posterior["mu"].mean(dim=["chain", "draw"]).values

        # Calculate expected covariance matrix (mean of the posterior distribution for 'cov')
        expected_cov = posterior["cov"].mean(dim=["chain", "draw"]).values

        return (
            pd.Series(
                expected_returns, index=self.assets_, name="Posterior Mean Returns"
            ),
            pd.DataFrame(expected_cov, index=self.assets_, columns=self.assets_),
        )
