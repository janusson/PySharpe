"""Unit tests for the metrics helper module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysharpe.metrics import (
    annualize_return,
    annualize_volatility,
    compute_returns,
    expected_return,
    sharpe_ratio,
)


def test_compute_returns_simple_series(sample_price_series):
    returns = compute_returns(sample_price_series)
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(sample_price_series) - 1
    np.testing.assert_allclose(
        returns.iloc[:3].values,
        np.array([0.02, -0.00980392, 0.01980198]),
        rtol=1e-6,
    )


def test_compute_returns_log_matches_manual(sample_price_series):
    log_returns = compute_returns(sample_price_series, method="log")
    manual = np.log(sample_price_series.iloc[1:].values / sample_price_series.iloc[:-1].values)
    np.testing.assert_allclose(log_returns.values, manual, rtol=1e-9)


def test_compute_returns_validates_method(sample_price_series):
    with pytest.raises(ValueError):
        compute_returns(sample_price_series, method="bogus")


def test_annualize_return_series(sample_price_series):
    returns = compute_returns(sample_price_series)
    result = annualize_return(returns, periods_per_year=252)
    manual = (1 + returns).prod() ** (252 / len(returns)) - 1
    assert pytest.approx(manual, rel=1e-9) == result


def test_annualize_return_dataframe(sample_price_frame):
    returns = compute_returns(sample_price_frame)
    result = annualize_return(returns, periods_per_year=252)
    assert isinstance(result, pd.Series)
    assert set(result.index) == {"AAA", "BBB", "CCC"}


def test_annualize_volatility_requires_samples(sample_price_series):
    with pytest.raises(ValueError):
        annualize_volatility(sample_price_series.iloc[:1])


def test_expected_return_matches_mean(sample_price_frame):
    returns = compute_returns(sample_price_frame)
    observed = expected_return(returns, periods_per_year=252)
    manual = returns.mean() * 252
    pd.testing.assert_series_equal(observed, manual)


def test_sharpe_ratio_handles_risk_free_rate(sample_price_series):
    returns = compute_returns(sample_price_series)
    sharpe = sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)
    ann_return = annualize_return(returns, periods_per_year=252)
    ann_vol = annualize_volatility(returns, periods_per_year=252)
    expected = (ann_return - 0.02) / ann_vol
    assert pytest.approx(expected, rel=1e-9) == sharpe


def test_sharpe_ratio_raises_when_vol_zero():
    returns = pd.Series([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        sharpe_ratio(returns)

