"""
Tests for the analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from pysharpe.analysis.scoring import (
    technical_score,
    dividend_score,
    composite_score,
    validate_data,
    validate_weights
)
from pysharpe.analysis.backtest import (
    prepare_backtest_data,
    optimize_portfolio,
    simulate_returns
)

def test_technical_score():
    """Test technical scoring function."""
    # Test normal case
    score = technical_score(100, 90, 110, 95, 15)
    assert 0 <= score <= 1
    
    # Test edge cases
    with pytest.raises(ValueError):
        technical_score(-100, 90, 110, 95, 15)
    with pytest.raises(ValueError):
        technical_score(100, -90, 110, 95, 15)
    with pytest.raises(ValueError):
        technical_score(100, 90, -110, 95, 15)
    
    # Test perfect score case
    perfect = technical_score(100, 100, 100, 100, 10)
    assert perfect <= 1
    
    # Test extreme P/E ratio
    high_pe = technical_score(100, 90, 110, 95, 100)
    low_pe = technical_score(100, 90, 110, 95, 5)
    assert high_pe < low_pe  # Lower P/E should score better
    
    # Test price relative to moving averages
    above_sma = technical_score(120, 90, 100, 95, 15)
    below_sma = technical_score(80, 90, 100, 95, 15)
    assert above_sma < below_sma  # Price below SMA should score better

def test_dividend_score():
    """Test dividend scoring function."""
    # Test normal case
    score = dividend_score(0.04, 0.6, 8, 0.05, 3)
    assert 0 <= score <= 1
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        dividend_score(-0.04, 0.6, 8, 0.05)
    
    # Test yield caps
    high_yield = dividend_score(0.15, 0.6, 8, 0.05, 3)  # 15% yield
    normal_yield = dividend_score(0.04, 0.6, 8, 0.05, 3)  # 4% yield
    excessive_yield = dividend_score(0.25, 0.6, 8, 0.05, 3)  # 25% yield
    assert high_yield >= normal_yield  # Higher yield should score better
    assert high_yield == excessive_yield  # But should cap at maximum
    
    # Test payout ratio impact
    low_payout = dividend_score(0.04, 0.3, 8, 0.05, 3)
    high_payout = dividend_score(0.04, 0.9, 8, 0.05, 3)
    assert low_payout > high_payout  # Lower payout ratio should score better
    
    # Test consecutive increases impact
    long_history = dividend_score(0.04, 0.6, 15, 0.05, 3)
    short_history = dividend_score(0.04, 0.6, 3, 0.05, 3)
    assert long_history > short_history  # Longer dividend history should score better

def test_composite_score():
    """Test composite scoring function."""
    # Test normal case
    score = composite_score(0.5, 0.5)
    assert 0 <= score <= 1
    
    # Test invalid weights
    with pytest.raises(ValueError):
        composite_score(0.7, 0.7)  # Sum > 1

def test_validate_weights():
    """Test weight validation."""
    valid_weights = {'a': 0.5, 'b': 0.5}
    invalid_weights = {'a': 0.7, 'b': 0.7}
    
    # Test valid case
    validate_weights(valid_weights, ['a', 'b'])
    
    # Test invalid cases
    with pytest.raises(ValueError):
        validate_weights(invalid_weights, ['a', 'b'])
    
    with pytest.raises(ValueError):
        validate_weights({'a': 0.5}, ['a', 'b'])

def test_backtest_preparation():
    """Test backtest data preparation."""
    df = pd.DataFrame({
        'Ticker': ['A', 'B'],
        'CompositeScore': [0.5, 0.7]
    })
    
    mu, cov = prepare_backtest_data(df)
    
    assert isinstance(mu, pd.Series)
    assert isinstance(cov, pd.DataFrame)
    assert len(mu) == len(df)
    assert cov.shape == (len(df), len(df))

def test_portfolio_optimization():
    """Test portfolio optimization."""
    returns = pd.Series({'A': 0.1, 'B': 0.2})
    cov = pd.DataFrame([[0.1, 0.0], [0.0, 0.1]], index=['A', 'B'], columns=['A', 'B'])
    
    weights = optimize_portfolio(returns, cov)
    
    assert isinstance(weights, dict)
    assert np.isclose(sum(weights.values()), 1.0)
    assert all(w >= 0 for w in weights.values())