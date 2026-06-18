#!/usr/bin/env python3
"""Stress test the Bayesian portfolio optimization algorithms and memory management.

Introduces additional CAD-denominated dividend and broad-market ETFs via proxies
to simulate a larger, more complex portfolio environment.
"""

import logging
import sys
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# Add src to sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pysharpe.optimization.bayesian import BayesianOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("bayesian_stress_test")

# Proxies defined from docs/proxies.md to represent CAD-denominated and global ETFs
# with sufficient historical data for robust Bayesian sampling.
PROXY_MAPPING = {
    "VFV.TO": "VOO",  # U.S. Large Cap
    "INAI.TO": "QQQ",  # NASDAQ 100
    "VCN.TO": "XIU.TO",  # Canada All Cap
    "VIDY.TO": "VIGI",  # Intl. Div. Apprec.
    "VUDV.TO": "VYM",  # U.S. High Yield
    "FCUQ.TO": "QUAL",  # U.S. Quality
    "FCUV.TO": "VLUE",  # U.S. Value
    "VMO.TO": "MTUM",  # Global Momentum
    "ZGLD.TO": "GLD",  # Physical Gold
}


def fetch_data() -> pd.DataFrame:
    """Fetch historical data for the proxy tickers."""
    tickers = list(PROXY_MAPPING.values())
    logger.info(f"Downloading 10-year price data for proxies: {tickers}")

    # Download 10 years of data to stress the model
    data = yf.download(tickers, start="2014-01-01", end="2024-01-01", progress=False)[
        "Close"
    ]

    # Drop rows with NaNs to ensure clean data for PyMC
    data = data.dropna()

    # Calculate daily log returns
    import numpy as np

    returns = np.log(data / data.shift(1)).dropna()

    # Rename columns back to target ETF names for the output
    reverse_mapping = {v: k for k, v in PROXY_MAPPING.items()}
    returns = returns.rename(columns=reverse_mapping)

    logger.info(f"Prepared return data shape: {returns.shape}")
    return returns


def run_stress_test():
    """Run the Bayesian Optimizer and perform a Monte Carlo correlation stress test."""
    returns = fetch_data()

    # We will resample to monthly returns to speed up the test and reduce trace memory size
    # MCMC with daily data across 10 years for 8 assets takes significant time.
    monthly_returns = returns.resample("ME").sum()
    logger.info(f"Resampled to monthly returns. New shape: {monthly_returns.shape}")

    optimizer = BayesianOptimizer(random_seed=42)

    tracemalloc.start()

    logger.info("Initiating MCMC sampling. This stresses both CPU and memory...")
    try:
        # Using a moderately high number of draws/tunes to stress the system
        # without causing an outright timeout.
        optimizer.fit_returns_model(
            monthly_returns,
            draws=400,
            tune=200,
            target_accept=0.9,
            cores=1,  # Limit cores to ensure stability in constrained environments
            chains=2,
        )

        current, peak = tracemalloc.get_traced_memory()
        logger.info(
            f"Memory Usage - Current: {current / 10**6:.2f} MB, Peak: {peak / 10**6:.2f} MB"
        )

        expected_returns, expected_cov = optimizer.get_posterior_estimates()

        assets = list(expected_cov.columns)
        if "VFV.TO" not in assets or "INAI.TO" not in assets:
            logger.error("Required assets VFV.TO and INAI.TO not found in data.")
            return

        idx1 = assets.index("VFV.TO")
        idx2 = assets.index("INAI.TO")

        # Decompose covariance to correlation and standard deviations
        cov_matrix = expected_cov.values
        stdevs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(stdevs, stdevs)

        base_corr = corr_matrix[idx1, idx2]
        logger.info(f"Base correlation between VFV.TO and INAI.TO: {base_corr:.4f}")

        # Monte Carlo Simulation
        num_simulations = 1000
        sharpe_ratios = []

        logger.info(
            f"Running {num_simulations} Monte Carlo simulations varying correlation by ±20%..."
        )

        for _ in range(num_simulations):
            # Sample a variation between -20% and +20%
            variation = np.random.uniform(-0.20, 0.20)
            new_corr = base_corr * (1 + variation)

            # Ensure valid correlation bounds [-1, 1]
            new_corr = max(-1.0, min(1.0, new_corr))

            # Create a modified correlation matrix
            mod_corr_matrix = corr_matrix.copy()
            mod_corr_matrix[idx1, idx2] = new_corr
            mod_corr_matrix[idx2, idx1] = new_corr

            # Ensure the matrix remains positive semi-definite (basic check/fix by clipping eigenvalues)
            eigvals, eigvecs = np.linalg.eigh(mod_corr_matrix)
            eigvals[eigvals < 0] = 1e-8
            mod_corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Reconstruct covariance matrix
            mod_cov_matrix = mod_corr_matrix * np.outer(stdevs, stdevs)

            # Optimize with the modified covariance matrix
            n_assets = len(assets)

            def objective(weights):
                # Using annualized metrics (assuming monthly data * 12)
                p_return = np.sum(expected_returns.values * weights) * 12
                p_vol = np.sqrt(np.dot(weights.T, np.dot(mod_cov_matrix * 12, weights)))
                sharpe = p_return / p_vol if p_vol > 0 else 0
                return -sharpe

            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            bounds = tuple((0.0, 1.0) for _ in range(n_assets))
            initial_guess = np.array([1.0 / n_assets] * n_assets)

            res = minimize(
                objective,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if res.success:
                # Store the positive Sharpe ratio
                sharpe_ratios.append(-res.fun)

        logger.info(
            f"Completed simulations. Mean Sharpe: {np.mean(sharpe_ratios):.4f}, Std: {np.std(sharpe_ratios):.4f}"
        )

        # Plotting the distribution
        plt.figure(figsize=(10, 6))
        plt.hist(sharpe_ratios, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        plt.title(
            "Distribution of Optimal Sharpe Ratios\n(VFV.TO - INAI.TO Correlation varied by ±20%)"
        )
        plt.xlabel("Sharpe Ratio")
        plt.ylabel("Frequency")
        plt.axvline(
            np.mean(sharpe_ratios),
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {np.mean(sharpe_ratios):.2f}",
        )
        plt.legend()

        # Save the plot
        output_dir = ROOT / "data" / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "bayesian_mc_sharpe_distribution.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Stress test completed successfully. Plot saved to {plot_path}")

    except Exception as e:
        logger.error(f"Stress test failed: {e}", exc_info=True)
    finally:
        tracemalloc.stop()


if __name__ == "__main__":
    run_stress_test()
