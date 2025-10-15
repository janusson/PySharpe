"""
Backtesting module for scoring-based portfolio strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pypfopt import EfficientFrontier, risk_models, expected_returns

def prepare_backtest_data(
    df: pd.DataFrame,
    score_col: str = 'CompositeScore',
    scaling_factor: float = 0.05
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Prepare data for backtesting by converting scores to expected returns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing scores and tickers
    score_col : str
        Name of the score column to use
    scaling_factor : float
        Factor to scale scores into expected returns

    Returns
    -------
    tuple
        (expected returns Series, covariance matrix DataFrame)
    """
    tickers = df['Ticker'].tolist()
    
    # Convert scores to expected returns
    mu = pd.Series(df[score_col] * scaling_factor, index=tickers)
    
    # Create a simple covariance matrix (can be replaced with historical data)
    n = len(tickers)
    random_matrix = np.random.rand(n, n)
    cov_matrix = pd.DataFrame(
        (random_matrix + random_matrix.T) / 2,
        index=tickers,
        columns=tickers
    )
    # Ensure positive definiteness
    cov_matrix += n * np.eye(n)
    
    return mu, cov_matrix

def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame
) -> Dict[str, float]:
    """
    Optimize portfolio weights using the efficient frontier.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns for each asset
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns

    Returns
    -------
    dict
        Optimized portfolio weights
    """
    ef = EfficientFrontier(expected_returns, cov_matrix)
    ef.max_sharpe()
    return ef.clean_weights()

def simulate_returns(
    df: pd.DataFrame,
    weights: Dict[str, float],
    cov_matrix: pd.DataFrame,
    periods: int = 12,
    initial_value: float = 100000,
    scaling_factor: float = 0.05
) -> List[float]:
    """
    Simulate portfolio returns using a multivariate normal distribution.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing scores and tickers
    weights : dict
        Portfolio weights for each ticker
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns
    periods : int
        Number of periods to simulate
    initial_value : float
        Initial portfolio value
    scaling_factor : float
        Factor to scale scores into expected returns

    Returns
    -------
    list
        Simulated portfolio values over time
    """
    tickers = df['Ticker'].tolist()
    
    # Create expected returns from composite scores
    mu = df['CompositeScore'] * scaling_factor
    
    # Convert weights dict to array
    weights_arr = np.array([weights[ticker] for ticker in tickers])
    
    # Simulate returns
    simulated_returns = np.random.multivariate_normal(
        mu,
        cov_matrix.values,
        periods
    )
    
    # Calculate portfolio returns
    port_returns = simulated_returns.dot(weights_arr)
    
    # Calculate portfolio values
    portfolio_values = [initial_value]
    for r in port_returns:
        portfolio_values.append(portfolio_values[-1] * (1 + r))
    
    return portfolio_values