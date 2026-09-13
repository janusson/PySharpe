"""Tests for the analysis module (backtest, scoring).

.. note::

    **Canadian ETF Mandate** — PySharpe targets broad-market, CAD-
    denominated index ETFs (e.g., VFV, VDY, QQC).  :func:`technical_score`
    and :func:`dividend_score` are deprecated single-stock scoring
    functions that fall outside the ETF scope.  These tests validate the
    scoring math but the functions should not be used for portfolio
    allocation decisions.

    For ETF-level evaluation, use the VA allocation engine:
    :func:`pysharpe.execution.allocator.score_opportunities`.
"""

from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pandas as pd
import pytest

from pysharpe.analysis.backtest import (
    optimize_portfolio,
    prepare_backtest_data,
    simulate_returns,
)
from pysharpe.analysis.benchmarks import (
    build_benchmark_characteristics,
    fetch_benchmark_metrics,
)
from pysharpe.analysis.scoring import (
    composite_score,
    dividend_score,
    technical_score,
    validate_weights,
)
from pysharpe.analysis.visualization import (
    plot_backtest_results,
    plot_score_comparison,
    plot_score_distribution,
)
from pysharpe.optimization.expected_returns import shrinkage_expected_return
from pysharpe.optimization.tax_location import (
    AssetLocationEngine,
    TaxProfile,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_plots():
    yield
    plt.close("all")


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
    valid_weights = {"a": 0.5, "b": 0.5}
    invalid_weights = {"a": 0.7, "b": 0.7}

    # Test valid case
    validate_weights(valid_weights, ["a", "b"])

    # Test invalid cases
    with pytest.raises(ValueError):
        validate_weights(invalid_weights, ["a", "b"])

    with pytest.raises(ValueError):
        validate_weights({"a": 0.5}, ["a", "b"])


def test_backtest_preparation():
    """Test backtest data preparation."""
    df = pd.DataFrame({"Ticker": ["A", "B"], "CompositeScore": [0.5, 0.7]})

    mu, cov = prepare_backtest_data(df)

    assert isinstance(mu, pd.Series)
    assert isinstance(cov, pd.DataFrame)
    assert len(mu) == len(df)
    assert cov.shape == (len(df), len(df))


def test_portfolio_optimization():
    """Test portfolio optimization."""
    returns = pd.Series({"A": 0.1, "B": 0.2})
    cov = pd.DataFrame([[0.1, 0.0], [0.0, 0.1]], index=["A", "B"], columns=["A", "B"])

    weights = optimize_portfolio(returns, cov)

    assert isinstance(weights, dict)
    assert np.isclose(sum(weights.values()), 1.0)
    assert all(w >= 0 for w in weights.values())


def test_prepare_backtest_data_is_reproducible():
    df = pd.DataFrame(
        {
            "Ticker": ["A", "B"],
            "CompositeScore": [0.5, 0.7],
        }
    )

    _, cov_a = prepare_backtest_data(df, random_state=123)
    _, cov_b = prepare_backtest_data(df, random_state=123)
    pd.testing.assert_frame_equal(cov_a, cov_b)


def test_simulate_returns_is_reproducible():
    df = pd.DataFrame(
        {
            "Ticker": ["A", "B"],
            "CompositeScore": [0.05, 0.07],
        }
    )
    weights = {"A": 0.6, "B": 0.4}
    cov = pd.DataFrame(
        [[0.1, 0.02], [0.02, 0.2]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    results_a = simulate_returns(
        df,
        weights,
        cov,
        periods=4,
        initial_value=1_000,
        random_state=123,
    )
    results_b = simulate_returns(
        df,
        weights,
        cov,
        periods=4,
        initial_value=1_000,
        random_state=123,
    )

    assert results_a == results_b
    assert len(results_a) == 5


def test_fetch_benchmark_metrics_empty():
    """Test with empty ticker list."""
    df = fetch_benchmark_metrics([], "2023-01-01", "2023-12-31")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "Ticker",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
    ]
    assert len(df) == 0


def test_fetch_benchmark_metrics_mocked(monkeypatch: pytest.MonkeyPatch):
    """Test with mocked price data."""
    mock_fetcher = MagicMock()
    veqt_data = pd.DataFrame(
        {"Close": [10.0, 10.1, 10.2]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
    )
    mock_fetcher.fetch_history.return_value = veqt_data
    monkeypatch.setattr(
        "pysharpe.analysis.benchmarks.apply_fx_conversion", lambda df, **kwargs: df
    )
    monkeypatch.setattr(
        "pysharpe.analysis.benchmarks.DuckDBCachedPriceFetcher", lambda _: mock_fetcher
    )

    df = fetch_benchmark_metrics(["VEQT.TO"], "2023-01-01", "2023-01-03")
    assert len(df) == 1
    assert df.iloc[0]["Ticker"] == "VEQT.TO"
    assert "Annualized Return" in df.columns
    assert "Annualized Volatility" in df.columns
    assert "Sharpe Ratio" in df.columns
    assert df.iloc[0]["Annualized Return"] > 0


def _mock_benchmark_fetcher(monkeypatch, prices):
    """Patch the benchmark fetcher to return *prices* (no network)."""
    mock_fetcher = MagicMock()
    mock_fetcher.fetch_history.return_value = prices
    monkeypatch.setattr(
        "pysharpe.analysis.benchmarks.apply_fx_conversion", lambda df, **kwargs: df
    )
    monkeypatch.setattr(
        "pysharpe.analysis.benchmarks.DuckDBCachedPriceFetcher", lambda _: mock_fetcher
    )
    return mock_fetcher


def test_fetch_benchmark_metrics_harmonized_joint_shrinkage_and_drag(monkeypatch):
    """Benchmarks are shrunk jointly with the universe and hit with MER+tax drag.

    The harmonized path must (1) evaluate the benchmark alongside the asset
    universe so Bayes-Stein shrinkage targets the same grand mean, and
    (2) route the shrunk return through the AssetLocationEngine so MER and
    income-tax drag are deducted before the Sharpe ratio is recomputed.
    """
    n = 40
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    universe = pd.DataFrame(
        {
            "A.TO": 100 * (1 + rng.normal(0.0004, 0.010, n)).cumprod(),
            "B.TO": 100 * (1 + rng.normal(0.0012, 0.011, n)).cumprod(),
        },
        index=idx,
    )
    bm = pd.DataFrame(
        {"Close": 100 * (1 + rng.normal(0.0020, 0.013, n)).cumprod()}, index=idx
    )
    _mock_benchmark_fetcher(monkeypatch, bm)

    profile = TaxProfile(marginal_tax_rate=0.45)
    df = fetch_benchmark_metrics(
        ["VEQT.TO"],
        "2023-01-01",
        "2023-02-28",
        reference_prices=universe,
        tax_profile=profile,
    )

    assert len(df) == 1
    row = df.iloc[0]

    combined = pd.concat(
        [universe, bm.rename(columns={"Close": "VEQT.TO"})], axis=1
    ).dropna()
    mu = shrinkage_expected_return(combined, shrinkage_floor=0.3)["VEQT.TO"]
    char = build_benchmark_characteristics("VEQT.TO")
    adjusted = AssetLocationEngine(profile).compute_tax_adjusted_return(
        float(mu), char, "NON_REG"
    )
    vol = combined.pct_change().dropna()["VEQT.TO"].std() * np.sqrt(252)

    assert row["Annualized Return"] == pytest.approx(adjusted)
    assert row["Annualized Volatility"] == pytest.approx(vol)
    assert row["Sharpe Ratio"] == pytest.approx((adjusted - 0.02) / vol)
    # The drag model must penalize: MER (decimal) + income-tax drag.
    assert row["Annualized Return"] < float(mu)


def test_fetch_benchmark_metrics_shrinks_toward_joint_grand_mean(monkeypatch):
    """Isolation edge case: benchmarks must shrink toward the joint grand mean.

    A benchmark with an extreme raw return is pulled toward the cross-
    sectional grand mean of the combined universe when evaluated jointly.
    """
    n = 40
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(13)
    universe = pd.DataFrame(
        {
            "A.TO": 100 * (1 + rng.normal(0.0005, 0.010, n)).cumprod(),
            "B.TO": 100 * (1 + rng.normal(0.0006, 0.010, n)).cumprod(),
        },
        index=idx,
    )
    bm = pd.DataFrame(
        {"Close": 100 * (1 + rng.normal(0.0060, 0.010, n)).cumprod()}, index=idx
    )
    _mock_benchmark_fetcher(monkeypatch, bm)

    profile = TaxProfile(marginal_tax_rate=0.45)
    df = fetch_benchmark_metrics(
        ["VEQT.TO"],
        "2023-01-01",
        "2023-02-28",
        reference_prices=universe,
        tax_profile=profile,
    )
    assert len(df) == 1

    combined = pd.concat(
        [universe, bm.rename(columns={"Close": "VEQT.TO"})], axis=1
    ).dropna()
    mu = shrinkage_expected_return(combined, shrinkage_floor=0.3)
    raw = combined.pct_change().dropna().mean() * 252
    grand = float(raw.mean())
    bm_raw = float(raw["VEQT.TO"])
    bm_shrunk = float(mu["VEQT.TO"])

    assert bm_raw > grand  # sanity: the benchmark is the extreme asset
    assert abs(bm_shrunk - grand) <= abs(bm_raw - grand) + 1e-12


def test_fetch_benchmark_metrics_isolated_falls_back_to_raw_mean_with_drag(monkeypatch):
    """Isolated benchmark: shrinkage degenerates to the raw mean (no cross-
    section), but MER + income-tax drag still applies."""
    n = 30
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(11)
    bm = pd.DataFrame(
        {"Close": 100 * (1 + rng.normal(0.0010, 0.012, n)).cumprod()}, index=idx
    )
    _mock_benchmark_fetcher(monkeypatch, bm)

    profile = TaxProfile(marginal_tax_rate=0.45)
    df = fetch_benchmark_metrics(
        ["VEQT.TO"], "2023-01-01", "2023-02-28", tax_profile=profile
    )

    assert len(df) == 1
    row = df.iloc[0]
    # Single-asset shrinkage degenerates to the estimator's documented
    # fallback (geometric annualized mean) — recompute it through the same
    # function rather than assuming a specific formula.
    mu = shrinkage_expected_return(
        bm.rename(columns={"Close": "VEQT.TO"}), shrinkage_floor=0.3
    )
    char = build_benchmark_characteristics("VEQT.TO")
    adjusted = AssetLocationEngine(profile).compute_tax_adjusted_return(
        float(mu["VEQT.TO"]), char, "NON_REG"
    )
    assert row["Annualized Return"] == pytest.approx(adjusted)
    # MER is a decimal fraction (0.0017), never divided by 100.
    assert row["Annualized Return"] < float(mu["VEQT.TO"]) - char.mer


def test_benchmark_characteristics_defaults_are_decimal_and_sum_to_one():
    """Benchmark MERs are decimal fractions and income fractions sum to 1."""
    for ticker in ("VEQT.TO", "XEQT.TO", "VGRO.TO", "XGRO.TO", "VBAL.TO", "XBAL.TO"):
        char = build_benchmark_characteristics(ticker)
        assert 0.0 <= char.mer < 0.10
        total = (
            char.income_frac_interest
            + char.income_frac_eligible_dividends
            + char.income_frac_foreign_income
            + char.income_frac_capital_gains
        )
        assert total == pytest.approx(1.0)
        assert char.is_cad_wrapped_us_equity is True
    assert build_benchmark_characteristics("VEQT.TO").mer == pytest.approx(0.0017)


def test_plot_score_distribution():
    df = pd.DataFrame({"CompositeScore": [0.1, 0.5, 0.8, 0.9, 0.5]})
    plot_score_distribution(df, title="Test Dist")
    fig = plt.gcf()
    assert fig.axes[0].get_title() == "Test Dist"


def test_plot_score_comparison():
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "GOOGL"],
            "TechScore": [0.6, 0.8, 0.7],
            "DivScore": [0.2, 0.4, 0.0],
        }
    )
    plot_score_comparison(df)
    fig = plt.gcf()
    assert "TechScore vs DivScore" in fig.axes[0].get_title()


def test_plot_backtest_results():
    values = [1000.0, 1050.0, 1100.0, 1080.0]
    plot_backtest_results(values, periods=3, initial_value=1000.0)
    fig = plt.gcf()
    assert "Backtest: Portfolio Performance" in fig.axes[0].get_title()
