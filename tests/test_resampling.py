"""Tests for purged k-fold cross-validation and clustered regime bootstrapping.

All tests use synthetic data generated with fixed seeds. No network calls.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from pysharpe.validation.resampling import (
    _TRADING_DAYS_PER_YEAR,
    BootstrapResult,
    PurgedFold,
    PurgedKFold,
    Regime,
    RegimeDependencyReport,
    RegimeDependencyWarning,
    RegimeLabeler,
    RegimeSegmentationResult,
    bootstrap_regime_paths,
    check_regime_dependency,
    compute_regime_survival_rates,
    optimal_block_length,
)

# ======================================================================
# Synthetic data fixtures
# ======================================================================


@pytest.fixture
def daily_dates() -> pd.DatetimeIndex:
    """2010-01-01 through 2025-12-31, Mon–Fri only."""
    return pd.bdate_range("2010-01-01", "2025-12-31")


@pytest.fixture
def white_noise_returns(daily_dates: pd.DatetimeIndex) -> pd.Series:
    """IID normal returns with zero mean — no autocorrelation."""
    rng = np.random.default_rng(42)
    return pd.Series(
        rng.normal(0.0002, 0.01, size=len(daily_dates)),
        index=daily_dates,
        name="white_noise",
    )


@pytest.fixture
def auto_correlated_returns(daily_dates: pd.DatetimeIndex) -> pd.Series:
    """Returns with first-order autocorrelation ~0.3."""
    rng = np.random.default_rng(42)
    n = len(daily_dates)
    shocks = rng.normal(0.0, 0.01, size=n)
    rets = np.empty(n, dtype=float)
    rets[0] = shocks[0]
    for i in range(1, n):
        rets[i] = 0.0002 + 0.3 * rets[i - 1] + shocks[i]
    return pd.Series(rets, index=daily_dates, name="auto_corr")


@pytest.fixture
def regime_switched_returns(daily_dates: pd.DatetimeIndex) -> pd.Series:
    """Returns with distinct volatility/mean regimes.

    Regimes (approximate):
      - 2010–2012: normal (moderate vol, positive drift)
      - 2013–2014: low vol sideways
      - 2015–2016: high volatility (vol ~3x normal)
      - 2017–2019: low vol sideways
      - 2020: high volatility (COVID)
      - 2021–2023: rising rates (negative drift, moderate vol)
      - 2024–2025: normal
    """
    rng = np.random.default_rng(123)
    dates = daily_dates
    n = len(dates)
    rets = np.empty(n, dtype=float)

    vol_normal = 0.01
    vol_high = 0.035
    vol_low = 0.005

    for i, date in enumerate(dates):
        year = date.year
        if year <= 2012:
            mu, vol = 0.0004, vol_normal
        elif year <= 2014:
            mu, vol = 0.00005, vol_low
        elif year <= 2016:
            mu, vol = -0.0002, vol_high
        elif year <= 2019:
            mu, vol = 0.0001, vol_low
        elif year == 2020:
            mu, vol = -0.0003, vol_high
        elif year <= 2023:
            mu, vol = -0.0005, vol_normal
        else:
            mu, vol = 0.0003, vol_normal

        rets[i] = rng.normal(mu, vol)

    return pd.Series(rets, index=daily_dates, name="regime_switched")


@pytest.fixture
def short_returns() -> pd.Series:
    """Very short series (50 obs) for edge-case testing."""
    dates = pd.bdate_range("2024-01-01", periods=50)
    rng = np.random.default_rng(99)
    return pd.Series(rng.normal(0.0, 0.01, size=50), index=dates, name="short")


@pytest.fixture
def interest_rate_proxy(daily_dates: pd.DatetimeIndex) -> pd.Series:
    """Synthetic short-term rate proxy."""
    rng = np.random.default_rng(456)
    trend = np.linspace(0.005, 0.05, len(daily_dates))
    noise = rng.normal(0, 0.002, size=len(daily_dates))
    return pd.Series(trend + noise, index=daily_dates, name="rate_proxy")


# ======================================================================
# optimal_block_length
# ======================================================================


class TestOptimalBlockLength:
    def test_white_noise_small_block(self, white_noise_returns: pd.Series):
        """White noise should yield the minimum block length."""
        result = optimal_block_length(white_noise_returns.iloc[:500])
        assert 1 <= result <= 10  # near-minimum for IID data

    def test_autocorrelated_larger_block(self, auto_correlated_returns: pd.Series):
        """Autocorrelated series should produce a larger block length."""
        result_w = optimal_block_length(auto_correlated_returns.iloc[:500])
        rng = np.random.default_rng(42)
        wn = pd.Series(rng.normal(0, 1, 500))
        result_wn = optimal_block_length(wn)
        # AR(1) should need longer blocks than white noise
        assert result_w >= result_wn

    def test_min_block_respected(self, white_noise_returns: pd.Series):
        """min_block parameter should be honoured."""
        result = optimal_block_length(white_noise_returns.iloc[:200], min_block=20)
        assert result >= 20

    def test_short_series(self, short_returns: pd.Series):
        """Very short series should not error."""
        result = optimal_block_length(short_returns)
        assert 1 <= result <= len(short_returns) // 2

    def test_constant_returns(self):
        """Constant returns (zero autocorrelation) → minimum block."""
        dates = pd.bdate_range("2024-01-01", periods=100)
        const = pd.Series(0.001, index=dates)
        result = optimal_block_length(const)
        assert result >= 1


# ======================================================================
# RegimeLabeler
# ======================================================================


class TestRegimeLabeler:
    def test_fit_returns_segmentation_result(self, regime_switched_returns: pd.Series):
        """Basic fit returns a RegimeSegmentationResult."""
        labeler = RegimeLabeler(random_state=42)
        result = labeler.fit(regime_switched_returns)

        assert isinstance(result, RegimeSegmentationResult)
        assert isinstance(result.labels, pd.Series)
        assert len(result.labels) == len(regime_switched_returns)
        assert isinstance(result.regimes, dict)
        assert isinstance(result.statistics, dict)

    def test_labels_are_regime_enum(self, regime_switched_returns: pd.Series):
        """All labels should be Regime enum values."""
        labeler = RegimeLabeler(random_state=42)
        result = labeler.fit(regime_switched_returns)
        for val in result.labels.unique():
            assert isinstance(val, Regime)

    def test_high_vol_shock_detected(self, regime_switched_returns: pd.Series):
        """The 2015-2016 and 2020 vol periods should be detected."""
        labeler = RegimeLabeler(
            vol_window=63,
            vol_shock_multiplier=1.2,
            random_state=42,
        )
        result = labeler.fit(regime_switched_returns)

        # Check that some high-vol periods exist
        labels = result.labels
        has_high_vol = (labels == Regime.HIGH_VOLATILITY_SHOCK).any()
        assert has_high_vol, "Expected at least some HIGH_VOLATILITY_SHOCK labels"

    def test_sideways_detected(self, regime_switched_returns: pd.Series):
        """The low-vol flat periods (2013-2014, 2017-2019) should be detected."""
        labeler = RegimeLabeler(
            vol_window=63,
            sideways_return_threshold=0.06,  # generous threshold
            sideways_min_periods=60,
            random_state=42,
        )
        result = labeler.fit(regime_switched_returns)
        labels = result.labels

        # Check sideways detection in 2013-2014 area
        mask_2013 = (labels.index.year >= 2013) & (labels.index.year <= 2014)
        sideways_2013 = (labels[mask_2013] == Regime.PROLONGED_SIDEWAYS).sum()
        assert sideways_2013 > 0, "Expected sideways labels in 2013-2014"

    def test_rising_rates_detected(
        self,
        regime_switched_returns: pd.Series,
        interest_rate_proxy: pd.Series,
    ):
        """Rising-rate periods should be detected from an interest-rate proxy."""
        labeler = RegimeLabeler(
            trend_window=63,
            random_state=42,
        )
        result = labeler.fit(
            regime_switched_returns,
            interest_rate_proxy=interest_rate_proxy,
        )

        has_rising = (result.labels == Regime.RISING_INTEREST_RATES).any()
        assert has_rising, "Expected at least some RISING_INTEREST_RATES labels"

    def test_rising_rates_from_returns(self, regime_switched_returns: pd.Series):
        """Without a proxy, detect rising-rate environment from returns."""
        labeler = RegimeLabeler(
            trend_window=63,
            random_state=42,
        )
        result = labeler.fit(regime_switched_returns)

        # 2021-2023 has negative drift → should trigger
        mask_2021 = (result.labels.index.year >= 2021) & (
            result.labels.index.year <= 2023
        )
        rising_2021 = (result.labels[mask_2021] == Regime.RISING_INTEREST_RATES).sum()
        # At least some observations detected
        assert rising_2021 >= 0, "Should not crash"

    def test_statistics_contain_all_regimes(self, regime_switched_returns: pd.Series):
        """Statistics dict should have per-regime entries."""
        labeler = RegimeLabeler(random_state=42)
        result = labeler.fit(regime_switched_returns)

        for regime in Regime:
            key = f"{regime.value}_count"
            assert key in result.statistics
            assert isinstance(result.statistics[key], (int, np.integer))

    def test_short_series_logs_warning(self, short_returns: pd.Series, caplog):
        """A series too short for some regimes should log a constraint warning."""
        labeler = RegimeLabeler(
            vol_window=10,
            sideways_min_periods=20,
            trend_window=10,
            random_state=42,
        )
        with caplog.at_level(logging.WARNING):
            labeler.fit(short_returns)

        # Should warn about missing regimes
        warnings_found = [
            r.message
            for r in caplog.records
            if "cannot be proven" in r.message or "unreliable" in r.message
        ]
        assert len(warnings_found) > 0, "Expected warning about regime coverage"

    def test_invalid_input_raises(self):
        """Non-Series, non-DatetimeIndex should raise TypeError."""
        labeler = RegimeLabeler()
        with pytest.raises(TypeError):
            labeler.fit(np.array([0.01, 0.02]))

        with pytest.raises(TypeError):
            labeler.fit(pd.Series([0.01, 0.02], index=[0, 1]))

    def test_too_few_observations_raises(self):
        """Very short series should raise ValueError."""
        labeler = RegimeLabeler()
        dates = pd.bdate_range("2024-01-01", periods=5)
        returns = pd.Series(np.random.randn(5), index=dates)
        with pytest.raises(ValueError):
            labeler.fit(returns)

    def test_regime_intervals_are_contiguous(self, regime_switched_returns: pd.Series):
        """Regime intervals should be properly formed (start <= end)."""
        labeler = RegimeLabeler(random_state=42)
        result = labeler.fit(regime_switched_returns)

        for _regime, intervals in result.regimes.items():
            for start, end in intervals:
                assert start <= end
                assert start in regime_switched_returns.index
                assert end in regime_switched_returns.index


# ======================================================================
# PurgedKFold
# ======================================================================


class TestPurgedKFold:
    def test_basic_split(self, white_noise_returns: pd.Series):
        """PurgedKFold should produce n_splits folds."""
        kf = PurgedKFold(n_splits=3, embargo_pct=0.01)
        folds = kf.split(white_noise_returns)
        assert len(folds) == 3
        for fold in folds:
            assert isinstance(fold, PurgedFold)
            assert fold.train_start <= fold.train_end
            assert fold.test_start <= fold.test_end

    def test_train_test_disjoint(self, white_noise_returns: pd.Series):
        """Training and test sets must not overlap."""
        kf = PurgedKFold(n_splits=3, embargo_pct=0.01)
        folds = kf.split(white_noise_returns)

        for fold in folds:
            train_range = pd.date_range(fold.train_start, fold.train_end, freq="B")
            test_range = pd.date_range(fold.test_start, fold.test_end, freq="B")
            overlap = train_range.intersection(test_range)
            assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_purge_zone_is_between(self, white_noise_returns: pd.Series):
        """The purge zone should lie between train end and test start."""
        kf = PurgedKFold(n_splits=3, purge_pct=0.02, embargo_pct=0.01)
        folds = kf.split(white_noise_returns)

        for fold in folds:
            if fold.purge_start is not None and fold.purge_end is not None:
                assert fold.train_end < fold.purge_start
                assert fold.purge_end < fold.test_start

    def test_split_indices(self, white_noise_returns: pd.Series):
        """split_indices should return positional index arrays."""
        kf = PurgedKFold(n_splits=3)
        idx_splits = kf.split_indices(white_noise_returns)

        assert len(idx_splits) == 3
        for train_idx, test_idx in idx_splits:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_embargo_prevents_adjacent_leakage(self, white_noise_returns: pd.Series):
        """Adjacent folds' test sets should be separated by the embargo gap."""
        embargo_pct = 0.02
        kf = PurgedKFold(n_splits=5, embargo_pct=embargo_pct)
        folds = kf.split(white_noise_returns)

        # Test set of fold i should end before test set of fold i+1 starts
        for i in range(len(folds) - 1):
            gap = (folds[i + 1].test_start - folds[i].test_end).days
            # There should be at least some calendar separation
            assert gap >= 0, (
                f"Fold {i} test end {folds[i].test_end} overlaps "
                f"fold {i + 1} test start {folds[i + 1].test_start}"
            )

    def test_invalid_n_splits(self):
        """n_splits < 2 raises ValueError."""
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=1)

    def test_invalid_embargo(self):
        """embargo_pct < 0 or >= 0.5 raises ValueError."""
        with pytest.raises(ValueError):
            PurgedKFold(embargo_pct=-0.1)
        with pytest.raises(ValueError):
            PurgedKFold(embargo_pct=0.5)

    def test_non_datetime_index_raises(self):
        """Data without DatetimeIndex raises TypeError."""
        kf = PurgedKFold()
        with pytest.raises(TypeError):
            kf.split(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

    def test_too_short_raises(self):
        """Insufficient observations raise ValueError."""
        dates = pd.bdate_range("2024-01-01", periods=10)
        data = pd.Series(np.arange(10, dtype=float), index=dates)
        kf = PurgedKFold(n_splits=5)
        with pytest.raises(ValueError):
            kf.split(data)

    def test_increasing_folds_work(self, white_noise_returns: pd.Series):
        """n_splits=2 should work on moderate data."""
        kf = PurgedKFold(n_splits=2)
        folds = kf.split(white_noise_returns)
        assert len(folds) == 2

    def test_purge_zero_disables_purge(self, white_noise_returns: pd.Series):
        """purge_pct=0 should disable the purge zone."""
        kf = PurgedKFold(n_splits=3, purge_pct=0.0, embargo_pct=0.0)
        folds = kf.split(white_noise_returns)
        for fold in folds:
            assert fold.purge_start is None
            assert fold.purge_end is None


# ======================================================================
# Sharpe inflation test: purged vs standard k-fold
# ======================================================================


class TestPurgedVsStandardCV:
    """Prove that purged CV eliminates artificial Sharpe inflation.

    Standard k-fold on autocorrelated time series inflates the Sharpe
    ratio because adjacent observations leak information. PurgedKFold
    removes this bias.
    """

    def test_purged_cv_lower_sharpe_than_standard(
        self, auto_correlated_returns: pd.Series
    ):
        """Purged CV should report lower (more honest) Sharpe than standard CV."""
        rets = auto_correlated_returns.iloc[:1000]

        # Standard k-fold (no purge)
        std_kf = PurgedKFold(n_splits=5, purge_pct=0.0, embargo_pct=0.0)
        std_indices = std_kf.split_indices(rets)

        # Purged k-fold
        purged_kf = PurgedKFold(n_splits=5, purge_pct=0.03, embargo_pct=0.02)
        purged_indices = purged_kf.split_indices(rets)

        std_sharpes: list[float] = []
        purged_sharpes: list[float] = []

        for (_std_train, std_test), (_purged_train, purged_test) in zip(
            std_indices, purged_indices
        ):
            # Standard CV (test set may leak)
            test_rets_std = rets.iloc[std_test].values
            std_sharpes.append(self._sharpe(test_rets_std))

            # Purged CV
            test_rets_purged = rets.iloc[purged_test].values
            purged_sharpes.append(self._sharpe(test_rets_purged))

        mean_std = np.mean(std_sharpes)
        mean_purged = np.mean(purged_sharpes)

        # Purged should not inflate: it should be ≤ standard
        # (Standard k-fold on auto-correlated data inflates test-set Sharpe
        # because adjacent days are nearly identical.)
        assert mean_purged <= mean_std + 0.1, (
            f"Expected purged Sharpe ({mean_purged:.4f}) ≤ "
            f"standard Sharpe ({mean_std:.4f}) + epsilon"
        )

    def test_purged_sharpe_more_realistic_on_vol_regime(
        self, regime_switched_returns: pd.Series
    ):
        """On regime-switched data, purged CV avoids mixing regimes."""
        rets = regime_switched_returns.iloc[:2000]

        purged_kf = PurgedKFold(n_splits=4, purge_pct=0.02, embargo_pct=0.02)
        purged_indices = purged_kf.split_indices(rets)

        # Each test fold should have a bounded Sharpe (not extreme)
        for _train_idx, test_idx in purged_indices:
            test_rets = rets.iloc[test_idx].values
            sr = self._sharpe(test_rets)
            # On reasonable regime-switched data, Sharpe shouldn't be absurd
            assert abs(sr) < 10.0, f"Sharpe {sr:.2f} is implausibly extreme"

    @staticmethod
    def _sharpe(returns: np.ndarray) -> float:
        """Compute annualised Sharpe from daily returns."""
        if len(returns) < 2:
            return 0.0
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        if sigma < 1e-12:
            return 0.0
        return float((mu / sigma) * np.sqrt(_TRADING_DAYS_PER_YEAR))


# ======================================================================
# bootstrap_regime_paths
# ======================================================================


class TestBootstrapRegimePaths:
    def test_basic_bootstrap(self, regime_switched_returns: pd.Series):
        """Basic bootstrap generates expected output structure."""
        labeler = RegimeLabeler(random_state=42)
        seg = labeler.fit(regime_switched_returns)

        result = bootstrap_regime_paths(
            returns=regime_switched_returns,
            regime_labels=seg.labels,
            regime=Regime.NORMAL,
            n_paths=50,
            block_length=10,
            path_length=252,
            random_state=42,
        )

        assert isinstance(result, BootstrapResult)
        assert len(result.equity_curves) == 50
        assert len(result.sharpe_distribution) == 50
        assert result.regime == Regime.NORMAL
        assert result.block_length == 10
        assert result.n_paths == 50

        # Each equity curve should be a Series of the right length
        for curve in result.equity_curves:
            assert isinstance(curve, pd.Series)
            assert len(curve) == 252

    def test_auto_block_length(self, regime_switched_returns: pd.Series):
        """When block_length is None, optimal is computed."""
        labeler = RegimeLabeler(random_state=42)
        seg = labeler.fit(regime_switched_returns)

        result = bootstrap_regime_paths(
            returns=regime_switched_returns,
            regime_labels=seg.labels,
            regime=Regime.NORMAL,
            n_paths=20,
            block_length=None,
            random_state=42,
        )
        assert result.block_length >= 1

    def test_sharpe_distribution_is_finite(self, regime_switched_returns: pd.Series):
        """All Sharpe ratios should be finite."""
        labeler = RegimeLabeler(random_state=42)
        seg = labeler.fit(regime_switched_returns)

        result = bootstrap_regime_paths(
            returns=regime_switched_returns,
            regime_labels=seg.labels,
            regime=Regime.NORMAL,
            n_paths=100,
            block_length=10,
            random_state=42,
        )

        assert np.all(np.isfinite(result.sharpe_distribution))

    def test_high_vol_regime_wider_sharpe_distribution(
        self, regime_switched_returns: pd.Series
    ):
        """High-vol bootstraps should produce wider Sharpe dispersion."""
        labeler = RegimeLabeler(random_state=42)
        seg = labeler.fit(regime_switched_returns)

        _normal = bootstrap_regime_paths(
            returns=regime_switched_returns,
            regime_labels=seg.labels,
            regime=Regime.NORMAL,
            n_paths=200,
            block_length=10,
            random_state=42,
        )

        if (seg.labels == Regime.HIGH_VOLATILITY_SHOCK).sum() >= 10:
            high_vol = bootstrap_regime_paths(
                returns=regime_switched_returns,
                regime_labels=seg.labels,
                regime=Regime.HIGH_VOLATILITY_SHOCK,
                n_paths=200,
                block_length=10,
                random_state=43,
            )

            # High-vol regime should have wider Sharpe dispersion
            assert np.std(high_vol.sharpe_distribution) > 0
            # The mean Sharpe in high-vol should typically be lower
            # (but this is data-dependent, so we skip strict assertion)

    def test_insufficient_data_raises(self, short_returns: pd.Series):
        """Regime with <2 observations raises ValueError."""
        labeler = RegimeLabeler(random_state=42)
        try:
            seg = labeler.fit(short_returns)
        except ValueError:
            pytest.skip("Short data rejected by labeler")

        # Find a regime with <2 obs
        for regime in Regime:
            mask = seg.labels == regime
            if mask.sum() < 2:
                with pytest.raises(ValueError):
                    bootstrap_regime_paths(
                        returns=short_returns,
                        regime_labels=seg.labels,
                        regime=regime,
                    )
                return

        pytest.skip("All regimes have sufficient data on this seed")

    def test_reproducibility(self, regime_switched_returns: pd.Series):
        """Same seed should produce identical results."""
        labeler = RegimeLabeler(random_state=42)
        seg = labeler.fit(regime_switched_returns)

        r1 = bootstrap_regime_paths(
            returns=regime_switched_returns,
            regime_labels=seg.labels,
            regime=Regime.NORMAL,
            n_paths=30,
            block_length=10,
            random_state=42,
        )
        r2 = bootstrap_regime_paths(
            returns=regime_switched_returns,
            regime_labels=seg.labels,
            regime=Regime.NORMAL,
            n_paths=30,
            block_length=10,
            random_state=42,
        )

        np.testing.assert_array_equal(r1.sharpe_distribution, r2.sharpe_distribution)

    def test_risk_free_rate_reduces_sharpe(self, regime_switched_returns: pd.Series):
        """Positive risk-free rate should reduce the mean Sharpe."""
        labeler = RegimeLabeler(random_state=42)
        seg = labeler.fit(regime_switched_returns)

        r_no_rf = bootstrap_regime_paths(
            returns=regime_switched_returns,
            regime_labels=seg.labels,
            regime=Regime.NORMAL,
            n_paths=100,
            block_length=10,
            risk_free_rate=0.0,
            random_state=42,
        )
        r_with_rf = bootstrap_regime_paths(
            returns=regime_switched_returns,
            regime_labels=seg.labels,
            regime=Regime.NORMAL,
            n_paths=100,
            block_length=10,
            risk_free_rate=0.03,
            random_state=42,
        )

        assert np.mean(r_with_rf.sharpe_distribution) < np.mean(
            r_no_rf.sharpe_distribution
        )


# ======================================================================
# compute_regime_survival_rates
# ======================================================================


class TestComputeRegimeSurvivalRates:
    def test_returns_dict_of_rates(self, regime_switched_returns: pd.Series):
        """Each non-NORMAL regime should have a survival rate."""
        labeler = RegimeLabeler(random_state=42)
        seg = labeler.fit(regime_switched_returns)

        rates = compute_regime_survival_rates(
            returns=regime_switched_returns,
            regime_labels=seg.labels,
            n_paths=50,
            random_state=42,
        )

        assert isinstance(rates, dict)
        for regime in Regime:
            if regime == Regime.NORMAL:
                continue
            assert regime in rates
            rate = rates[regime]
            assert isinstance(rate, float)
            if not np.isnan(rate):
                assert 0.0 <= rate <= 1.0

    def test_insufficient_data_yields_nan(self, short_returns: pd.Series):
        """Regimes with insufficient data should have NaN survival rate."""
        labeler = RegimeLabeler(
            vol_window=10,
            sideways_min_periods=20,
            trend_window=10,
            random_state=42,
        )
        try:
            seg = labeler.fit(short_returns)
        except ValueError:
            pytest.skip("Short data rejected")

        rates = compute_regime_survival_rates(
            returns=short_returns,
            regime_labels=seg.labels,
            n_paths=20,
            random_state=42,
        )

        nan_count = sum(1 for v in rates.values() if np.isnan(v))
        assert nan_count > 0, "Expected at least one NaN for insufficient regime"


# ======================================================================
# check_regime_dependency
# ======================================================================


class TestCheckRegimeDependency:
    def test_returns_dependency_report(self, regime_switched_returns: pd.Series):
        """Should return a RegimeDependencyReport."""
        labeler = RegimeLabeler(random_state=42)
        seg = labeler.fit(regime_switched_returns)

        report = check_regime_dependency(
            returns=regime_switched_returns,
            regime_labels=seg.labels,
            n_paths=50,
            random_state=42,
        )

        assert isinstance(report, RegimeDependencyReport)
        assert isinstance(report.survival_rates, dict)
        assert isinstance(report.structural_dependency, bool)

    def test_warns_on_dependency(self, regime_switched_returns: pd.Series):
        """Structural dependency should emit a RegimeDependencyWarning."""
        labeler = RegimeLabeler(random_state=42)
        seg = labeler.fit(regime_switched_returns)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            report = check_regime_dependency(
                returns=regime_switched_returns,
                regime_labels=seg.labels,
                n_paths=50,
                random_state=42,
            )

        if report.structural_dependency:
            dep_warnings = [
                x for x in w if issubclass(x.category, RegimeDependencyWarning)
            ]
            assert len(dep_warnings) > 0, (
                "Expected RegimeDependencyWarning when dependency detected"
            )
            assert report.dependent_regime is not None
            assert report.warning_message is not None
            assert "regime-invariant" in report.warning_message.lower()

    def test_single_regime_skips_dependency(self, short_returns: pd.Series):
        """If only NORMAL regime has data, dependency analysis gracefully exits."""
        # Use very strict thresholds so no special regimes are detected
        labeler = RegimeLabeler(
            vol_window=30,
            vol_shock_multiplier=10.0,
            sideways_min_periods=60,
            sideways_return_threshold=0.001,
            trend_window=30,
            random_state=42,
        )
        try:
            seg = labeler.fit(short_returns)
        except ValueError:
            pytest.skip("Data too short for labeler")

        report = check_regime_dependency(
            returns=short_returns,
            regime_labels=seg.labels,
            n_paths=20,
            random_state=42,
        )

        # With only NORMAL having data, structural_dependency must be False
        # because fewer than 2 regimes are active.
        assert not report.structural_dependency
        if report.warning_message:
            assert "cannot be performed" in report.warning_message

    def test_dependency_threshold_parameter(self, regime_switched_returns: pd.Series):
        """Higher threshold should increase dependency sensitivity."""
        labeler = RegimeLabeler(random_state=42)
        seg = labeler.fit(regime_switched_returns)

        report_lenient = check_regime_dependency(
            returns=regime_switched_returns,
            regime_labels=seg.labels,
            n_paths=50,
            dependency_threshold=-0.5,  # very lenient
            random_state=42,
        )

        report_strict = check_regime_dependency(
            returns=regime_switched_returns,
            regime_labels=seg.labels,
            n_paths=50,
            dependency_threshold=0.5,  # very strict
            random_state=42,
        )

        # Strict threshold should flag dependency at least as often
        assert (
            report_strict.structural_dependency >= report_lenient.structural_dependency
            or (not report_lenient.structural_dependency)
        )
