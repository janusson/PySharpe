"""
Scoring module for technical and fundamental analysis of securities.
Combines technical indicators and dividend metrics for comprehensive scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List

# Default weights for scoring functions
DEFAULT_TECH_WEIGHTS = {
    'P_YL': 0.3,
    'P_SMA200': 0.2,
    'P_ML3': 0.2,
    'PER': 0.3
}

DEFAULT_DIV_WEIGHTS = {
    'yield': 0.3,
    'payout': 0.2,
    'consec': 0.2,
    'growth': 0.2,
    'coverage': 0.1
}

def validate_weights(weights: Dict[str, float], expected_keys: List[str]) -> None:
    """Validate weight dictionary structure and values."""
    if not isinstance(weights, dict):
        raise TypeError("Weights must be provided as a dictionary")
    
    missing_keys = set(expected_keys) - set(weights.keys())
    if missing_keys:
        raise ValueError(f"Missing weights for: {missing_keys}")
    
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-10):
        raise ValueError("Weights must sum to 1.0")

def technical_score(
    P: float,
    YL: float,
    SMA200: float,
    ML3: float,
    PER: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate technical score based on price metrics.

    Parameters
    ----------
    P : float
        Current price
    YL : float
        52-week low
    SMA200 : float
        200-day simple moving average
    ML3 : float
        3-month low
    PER : float
        Price-to-earnings ratio
    weights : dict, optional
        Custom weights for each component

    Returns
    -------
    float
        Technical score between 0 and 1
    """
    # Input validation
    if any(val <= 0 for val in [P, YL, SMA200, ML3]):
        raise ValueError("Price values must be positive")
    
    weights = weights or DEFAULT_TECH_WEIGHTS
    validate_weights(weights, ['P_YL', 'P_SMA200', 'P_ML3', 'PER'])

    # Score calculations
    P_YL_score = min(P / YL, 2.0) / 2.0  # Normalize to [0, 1]
    P_SMA200_score = max(0, min(1, 1 - (P / SMA200 - 0.8)))  # Score higher when price is 20% below SMA200
    P_ML3_score = max(0, min(1, 1 - (P / ML3 - 0.9)))  # Score higher when price is 10% below 3-month low
    PER_score = 1 / (1 + max(0, PER) / 20)  # Normalize P/E ratio with decay

    total = (weights['P_YL'] * P_YL_score +
            weights['P_SMA200'] * P_SMA200_score +
            weights['P_ML3'] * P_ML3_score +
            weights['PER'] * PER_score)
    
    return float(total)

def dividend_score(
    div_yield: float,
    div_payout: float,
    consecutive_increases: int,
    div_growth: float,
    coverage: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate dividend score based on various dividend metrics.

    Parameters
    ----------
    div_yield : float
        Current dividend yield (as decimal)
    div_payout : float
        Dividend payout ratio (as decimal)
    consecutive_increases : int
        Number of consecutive years of dividend increases
    div_growth : float
        Dividend growth rate (as decimal)
    coverage : float, optional
        Dividend coverage ratio
    weights : dict, optional
        Custom weights for scoring components

    Returns
    -------
    float
        Dividend score between 0 and 1
    """
    # Input validation
    if div_yield < 0 or div_payout < 0 or consecutive_increases < 0 or div_growth < 0:
        raise ValueError("Dividend metrics must be non-negative")
    
    weights = weights or DEFAULT_DIV_WEIGHTS
    validate_weights(weights, ['yield', 'payout', 'consec', 'growth', 'coverage'])

    # Score calculations with improved normalization
    yield_norm = min(div_yield / 0.10, 1.0)  # Cap at 10% yield
    payout_norm = max(0, min(1, 1 - div_payout))  # Lower payout ratio is better
    consec_norm = min(consecutive_increases / 10, 1.0)  # Normalize to 10 years
    growth_norm = min(div_growth / 0.15, 1.0)  # Cap at 15% growth
    
    score = (weights['yield'] * yield_norm +
            weights['payout'] * payout_norm +
            weights['consec'] * consec_norm +
            weights['growth'] * growth_norm)
    
    if coverage is not None:
        coverage_norm = min(coverage / 5, 1.0)  # Normalize coverage to 5x
        score += weights['coverage'] * coverage_norm
    
    return float(score)

def composite_score(
    tech_score: float,
    div_score: float,
    alpha: float = 0.5,
    beta: float = 0.5
) -> float:
    """
    Combine technical and dividend scores with custom weights.

    Parameters
    ----------
    tech_score : float
        Technical analysis score
    div_score : float
        Dividend analysis score
    alpha : float
        Weight for technical score (default: 0.5)
    beta : float
        Weight for dividend score (default: 0.5)

    Returns
    -------
    float
        Composite score between 0 and 1
    """
    if not np.isclose(alpha + beta, 1.0):
        raise ValueError("Weights alpha and beta must sum to 1.0")

    if not (0.0 <= tech_score <= 1.0):
        raise ValueError("tech_score must be between 0 and 1.")
    if not (0.0 <= div_score <= 1.0):
        raise ValueError("div_score must be between 0 and 1.")
    if tech_score + div_score > 1.0:
        raise ValueError("Combined scores must not exceed 1.0 to preserve the weighted scale.")
    
    return float(alpha * tech_score + beta * div_score)

def validate_data(df: pd.DataFrame) -> None:
    """
    Validate input dataframe structure for scoring calculations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing security data
    
    Raises
    ------
    ValueError
        If required columns are missing
    """
    required_columns = [
        'Price', 'YearLow', 'SMA200', 'ThreeMonthLow', 'PER',
        'DividendYield', 'DividendPayout', 'ConsecutiveIncreases',
        'DividendGrowth', 'CoverageRatio'
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
