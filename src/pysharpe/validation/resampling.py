"""Purged k-fold cross-validation and clustered regime bootstrapping.

This module provides rigorous out-of-sample validation tools tailored to
financial time series, where standard i.i.d. cross-validation inflates
performance estimates through serial correlation leakage.

Key components
--------------
* ``PurgedKFold`` — time-series k-fold splitter with purge and embargo.
* ``RegimeLabeler`` — heuristic segmentation of market periods into
  HIGH_VOLATILITY_SHOCK, RISING_INTEREST_RATES, and PROLONGED_SIDEWAYS.
* ``bootstrap_regime_paths`` — non-parametric stationary block bootstrap
  that draws synthetic paths exclusively from specified regimes.
* ``compute_regime_survival_rates`` — per-regime Sharpe-ratio survival
  analysis for detecting structural dependency.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR: int = 252
_DEFAULT_N_BOOTSTRAP: int = 1_000
_DEFAULT_MIN_BLOCK: int = 5
_DEFAULT_N_SPLITS: int = 5
_DEFAULT_EMBARGO_PCT: float = 0.01  # fraction of total length
_MIN_OBSERVATIONS_PER_REGIME: int = 30

# Rolling-window defaults for regime detection
_VOL_WINDOW: int = 63  # ~3 months
_VOL_SHOCK_THRESHOLD: float = 1.5  # multiples of long-term median vol
_SIDEWAYS_RETURN_THRESHOLD: float = 0.03  # annualized return magnitude
_SIDEWAYS_MIN_PERIODS: int = 126  # ~6 months to qualify as sideways
_TREND_WINDOW: int = 63  # for rate regime detection


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Regime(Enum):
    """Macroeconomic regime categories for segmentation."""

    HIGH_VOLATILITY_SHOCK = "high_volatility_shock"
    RISING_INTEREST_RATES = "rising_interest_rates"
    PROLONGED_SIDEWAYS = "prolonged_sideways"
    NORMAL = "normal"


# ---------------------------------------------------------------------------
# Warning
# ---------------------------------------------------------------------------


class RegimeDependencyWarning(UserWarning):
    """Warn when a strategy's performance is structurally dependent on a single regime."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class RegimeSegmentationResult(NamedTuple):
    """Output of ``RegimeLabeler.fit()``.

    Attributes
    ----------
    labels:
        A Series with the same DatetimeIndex as the input, containing a
        ``Regime`` enum value for each observation.
    regimes:
        Mapping from each ``Regime`` to a list of ``(start, end)`` timestamp
        tuples representing contiguous intervals.
    statistics:
        Dictionary of per-regime descriptive statistics (count, mean return,
        vol, fraction of total observations).
    """

    labels: pd.Series
    regimes: dict[Regime, list[tuple[pd.Timestamp, pd.Timestamp]]]
    statistics: dict[str, Any]


@dataclass(frozen=True)
class PurgedFold:
    """A single purged train/test split.

    Attributes
    ----------
    train_start, train_end:
        Timestamp bounds of the training set (inclusive).
    test_start, test_end:
        Timestamp bounds of the test set (inclusive).
    purge_start, purge_end:
        Timestamp bounds of the purged (removed) observations between
        train and test.
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_start: pd.Timestamp | None = None
    purge_end: pd.Timestamp | None = None


@dataclass(frozen=True)
class RegimeDependencyReport:
    """Diagnostic report on regime dependency of a strategy.

    Attributes
    ----------
    survival_rates:
        Per-regime fraction of bootstrap paths with positive Sharpe ratio.
    structural_dependency:
        ``True`` when the strategy's positive Sharpe is entirely dependent
        on a single regime and fails in others.
    dependent_regime:
        The ``Regime`` on which performance depends (or ``None``).
    warning_message:
        Human-readable message explaining the dependency (or ``None``).
    """

    survival_rates: dict[Regime, float]
    structural_dependency: bool
    dependent_regime: Regime | None
    warning_message: str | None


@dataclass
class BootstrapResult:
    """Result of ``bootstrap_regime_paths``.

    Attributes
    ----------
    equity_curves:
        List of synthetic equity curves (pd.Series each).
    sharpe_distribution:
        Array of Sharpe ratios computed on each synthetic path.
    regime:
        The regime from which blocks were drawn.
    block_length:
        Optimal block length used for the bootstrap.
    n_paths:
        Number of bootstrap paths generated.
    """

    equity_curves: list[pd.Series]
    sharpe_distribution: np.ndarray
    regime: Regime
    block_length: int
    n_paths: int


# ---------------------------------------------------------------------------
# Block length selection (Politis–Romano / autocorrelation-based)
# ---------------------------------------------------------------------------


def optimal_block_length(
    returns: pd.Series,
    *,
    min_block: int = _DEFAULT_MIN_BLOCK,
    max_lag: int | None = None,
) -> int:
    """Select an optimal block length using the Politis–Romano (2004) method.

    The algorithm minimises the mean-squared error of the variance estimator
    under the stationary bootstrap. It uses the sample autocorrelations to
    estimate the optimal expected block length.

    Args:
        returns: A 1-D series of returns (decimal fractions).
        min_block: Minimum allowed block length.
        max_lag: Maximum lag for autocorrelation estimation. Defaults to
            ``min(252, len(returns) // 4)``.

    Returns:
        Optimal block length as an integer (clamped to ``[min_block, len(returns)]``).
    """
    n = len(returns)
    if n < min_block:
        return max(1, n)

    clean = returns.dropna().values
    if len(clean) < min_block:
        return max(1, len(clean))

    if max_lag is None:
        max_lag = min(_TRADING_DAYS_PER_YEAR, len(clean) // 4)
    max_lag = max(1, min(max_lag, len(clean) - 2))

    # De-mean
    x = clean - np.mean(clean)

    # Autocorrelations up to max_lag
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        if np.isnan(acf[lag]):
            acf[lag] = 0.0

    # Politis–Romano (2004) formula:
    # b_opt = ((3/2) * (sum_{k=-M}^{M} k * rho_k)^2 / sum_{k=-M}^{M} rho_k^2)^(1/3) * n^(1/3)
    # where rho_k is the sample autocorrelation at lag k.

    sum_sq_acf = np.sum(acf[1:] ** 2)

    if sum_sq_acf < 1e-12:
        # Nearly white noise — use minimum block length
        return min_block

    # Politis–White (2004) / Patton–Politis–White (2009) flat-top kernel variant
    # uses a simpler approximation: b_opt = (3/2)^(1/3) * (2 * rho / (1 - rho^2))^(2/3) * n^(1/3)
    # where rho is the first-order autocorrelation.

    rho1 = acf[1]
    if abs(rho1) < 0.01:
        return min_block

    # Use the simpler flat-top approximation
    ratio = (2.0 * abs(rho1)) / (1.0 - rho1**2)
    b_opt = int(
        np.ceil((1.5 ** (1.0 / 3.0)) * (ratio ** (2.0 / 3.0)) * (n ** (1.0 / 3.0)))
    )

    return max(min_block, min(b_opt, n // 2))


# ---------------------------------------------------------------------------
# Regime Labeler
# ---------------------------------------------------------------------------


class RegimeLabeler:
    """Segment historical market periods into macroeconomic regimes.

    The labeler uses heuristic thresholds on rolling volatility, cumulative
    returns, and (optionally) an interest-rate proxy to classify each
    observation into one of the enumerated regimes.

    Parameters
    ----------
    vol_window:
        Rolling window length for realised volatility estimation.
    vol_shock_multiplier:
        Multiplier above the long-term median volatility that triggers a
        ``HIGH_VOLATILITY_SHOCK`` classification.
    sideways_return_threshold:
        Annualised return magnitude below which a sufficiently long period
        is considered ``PROLONGED_SIDEWAYS``.
    sideways_min_periods:
        Minimum number of consecutive observations to label as sideways.
    trend_window:
        Rolling window used to detect persistent directional moves for
        ``RISING_INTEREST_RATES`` classification.
    random_state:
        Seed for reproducibility of any stochastic components (currently
        unused; reserved for future extensions).
    """

    def __init__(
        self,
        *,
        vol_window: int = _VOL_WINDOW,
        vol_shock_multiplier: float = _VOL_SHOCK_THRESHOLD,
        sideways_return_threshold: float = _SIDEWAYS_RETURN_THRESHOLD,
        sideways_min_periods: int = _SIDEWAYS_MIN_PERIODS,
        trend_window: int = _TREND_WINDOW,
        random_state: int | None = None,
    ) -> None:
        self.vol_window = vol_window
        self.vol_shock_multiplier = vol_shock_multiplier
        self.sideways_return_threshold = sideways_return_threshold
        self.sideways_min_periods = sideways_min_periods
        self.trend_window = trend_window
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        returns: pd.Series,
        *,
        interest_rate_proxy: pd.Series | None = None,
    ) -> RegimeSegmentationResult:
        """Segment the return series into macroeconomic regimes.

        Args:
            returns:
                Daily return series (decimal fractions) with a
                ``DatetimeIndex``.
            interest_rate_proxy:
                Optional daily series representing short-term interest rates
                or bond yields. When provided, trend detection on this series
                is used for the ``RISING_INTEREST_RATES`` regime instead of
                inferring from equity returns alone.

        Returns:
            A ``RegimeSegmentationResult`` containing labels, regime intervals,
            and per-regime statistics.

        Raises:
            TypeError: If *returns* is not a ``pd.Series``.
            ValueError: If the series has too few observations.
        """
        self._validate_input(returns)

        clean = returns.dropna()
        if clean.empty:
            raise ValueError("Return series contains no finite observations.")

        labels = pd.Series(Regime.NORMAL, index=returns.index, dtype=object)

        # --- Step 1: High-volatility shock detection ---
        labels = self._label_vol_shocks(clean, labels)

        # --- Step 2: Rising interest rates detection ---
        labels = self._label_rising_rates(clean, interest_rate_proxy, labels)

        # --- Step 3: Prolonged sideways detection (only on unlabelled) ---
        labels = self._label_sideways(clean, labels)

        # --- Build regime intervals ---
        regimes = self._build_regime_intervals(labels)

        # --- Compute statistics ---
        statistics = self._compute_statistics(clean, labels)

        # --- Data-quality warnings ---
        self._emit_coverage_warnings(regimes, clean.index)

        return RegimeSegmentationResult(
            labels=labels,
            regimes=regimes,
            statistics=statistics,
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(returns: pd.Series) -> None:
        if not isinstance(returns, pd.Series):
            raise TypeError("returns must be a pandas Series.")
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError("returns must have a DatetimeIndex.")
        if len(returns) < _MIN_OBSERVATIONS_PER_REGIME:
            raise ValueError(
                f"Need at least {_MIN_OBSERVATIONS_PER_REGIME} observations; "
                f"got {len(returns)}."
            )

    # ------------------------------------------------------------------
    # Regime detection helpers
    # ------------------------------------------------------------------

    def _label_vol_shocks(
        self,
        returns: pd.Series,
        labels: pd.Series,
    ) -> pd.Series:
        """Label periods where realised volatility exceeds the shock threshold."""
        if len(returns) < self.vol_window:
            return labels

        rolling_vol = returns.rolling(
            window=self.vol_window, min_periods=self.vol_window // 2
        ).std()
        long_term_median = rolling_vol.median()

        if pd.isna(long_term_median) or long_term_median < 1e-12:
            return labels

        threshold = long_term_median * self.vol_shock_multiplier
        shock_mask = rolling_vol > threshold

        # Fill small gaps with forward-fill to merge adjacent shock periods
        shock_mask = shock_mask.ffill(limit=10).fillna(False)

        labels.loc[shock_mask[shock_mask].index] = Regime.HIGH_VOLATILITY_SHOCK
        return labels

    def _label_rising_rates(
        self,
        returns: pd.Series,
        interest_rate_proxy: pd.Series | None,
        labels: pd.Series,
    ) -> pd.Series:
        """Detect periods of persistently rising rates.

        When an interest_rate_proxy is provided, trend direction is detected
        on that series. Otherwise, the method infers rising-rate environments
        from sustained negative equity returns attributable to rate pressure.
        """
        if interest_rate_proxy is not None:
            return self._label_rising_rates_from_proxy(
                returns, interest_rate_proxy, labels
            )
        return self._label_rising_rates_from_returns(returns, labels)

    def _label_rising_rates_from_proxy(
        self,
        returns: pd.Series,
        proxy: pd.Series,
        labels: pd.Series,
    ) -> pd.Series:
        """Use an explicit interest-rate proxy series."""
        aligned = proxy.reindex(returns.index)
        if aligned.dropna().empty:
            logger.warning(
                "Interest-rate proxy has no overlap with return index; "
                "falling back to return-based detection."
            )
            return self._label_rising_rates_from_returns(returns, labels)

        # Detect rising trend: slope of rolling linear regression
        if len(aligned.dropna()) < self.trend_window:
            return labels

        rolling_slope = self._rolling_trend(aligned.dropna(), self.trend_window)
        rising_mask = rolling_slope > 0

        # Require persistence: rising for at least trend_window // 2
        rising_mask = rising_mask.rolling(
            window=self.trend_window // 2, min_periods=1
        ).sum() > (self.trend_window // 4)

        idx = rising_mask[rising_mask].index
        # Only overwrite NORMAL observations
        mask_to_label = idx.intersection(labels[labels == Regime.NORMAL].index)
        labels.loc[mask_to_label] = Regime.RISING_INTEREST_RATES

        return labels

    def _label_rising_rates_from_returns(
        self,
        returns: pd.Series,
        labels: pd.Series,
    ) -> pd.Series:
        """Infer rising-rate regimes from sustained negative returns."""
        if len(returns) < self.trend_window:
            return labels

        cumulative = returns.cumsum()
        rolling_slope = self._rolling_trend(cumulative, self.trend_window)

        # Negative slope on cumulative returns → persistent drawdown
        falling_mask = rolling_slope < 0

        if len(returns) >= self.vol_window:
            rolling_vol = returns.rolling(
                window=self.vol_window, min_periods=self.vol_window // 2
            ).std()
            # Not a vol shock (those are already labelled)
            not_shock = rolling_vol <= rolling_vol.median() * self.vol_shock_multiplier
            falling_mask = falling_mask & not_shock

        falling_mask = falling_mask.rolling(
            window=self.trend_window // 2, min_periods=1
        ).sum() > (self.trend_window // 4)

        idx = falling_mask[falling_mask].index
        mask_to_label = idx.intersection(labels[labels == Regime.NORMAL].index)
        labels.loc[mask_to_label] = Regime.RISING_INTEREST_RATES

        return labels

    def _label_sideways(
        self,
        returns: pd.Series,
        labels: pd.Series,
    ) -> pd.Series:
        """Identify prolonged low-return, low-volatility periods."""
        if len(returns) < self.sideways_min_periods:
            return labels

        n = len(returns)
        cumulative = (1 + returns).cumprod()

        # Use a sliding window to detect flat segments
        half_win = self.sideways_min_periods // 2
        for start in range(0, n - self.sideways_min_periods + 1, half_win):
            end = start + self.sideways_min_periods
            segment = returns.iloc[start:end]

            # Annualised return magnitude over the segment
            total_return = cumulative.iloc[end - 1] / cumulative.iloc[start] - 1
            period_years = len(segment) / _TRADING_DAYS_PER_YEAR
            if period_years < 1e-6:
                continue
            ann_return = (1 + total_return) ** (1.0 / max(period_years, 1e-6)) - 1

            if abs(ann_return) < self.sideways_return_threshold:
                idx = segment.index.intersection(labels[labels == Regime.NORMAL].index)
                labels.loc[idx] = Regime.PROLONGED_SIDEWAYS

        return labels

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_trend(series: pd.Series, window: int) -> pd.Series:
        """Compute rolling linear regression slope."""
        if len(series) < window:
            return pd.Series(0.0, index=series.index)

        def _slope(y: np.ndarray) -> float:
            """OLS slope on the window data, skipping NaN entries."""
            # y is raw ndarray; its length equals the current window size
            # (which may be < `window` near the start of the series).
            mask = ~np.isnan(y)
            n_valid = mask.sum()
            if n_valid < max(2, len(y) // 2):
                return 0.0
            # Build x-values matching the actual window length
            x_local = np.arange(len(y), dtype=float)
            y_clean = y[mask]
            x_clean = x_local[mask]
            x_mean = x_clean.mean()
            denom = np.sum((x_clean - x_mean) ** 2)
            if denom < 1e-12:
                return 0.0
            y_mean = y_clean.mean()
            return float(np.sum((x_clean - x_mean) * (y_clean - y_mean)) / denom)

        return series.rolling(window=window, min_periods=window // 2).apply(
            _slope, raw=True
        )

    @staticmethod
    def _build_regime_intervals(
        labels: pd.Series,
    ) -> dict[Regime, list[tuple[pd.Timestamp, pd.Timestamp]]]:
        """Convert label series to contiguous intervals per regime."""
        regimes: dict[Regime, list[tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(
            list
        )

        for regime in Regime:
            mask = labels == regime
            if not mask.any():
                continue

            # Find contiguous blocks
            groups = (~mask).cumsum()[mask]
            for _, group in labels[mask].groupby(groups):
                if len(group) >= 2:
                    regimes[regime].append((group.index[0], group.index[-1]))

        return dict(regimes)

    @staticmethod
    def _compute_statistics(
        returns: pd.Series,
        labels: pd.Series,
    ) -> dict[str, Any]:
        """Compute per-regime descriptive statistics."""
        stats: dict[str, Any] = {"total_observations": len(labels)}

        common_idx = labels.index.intersection(returns.index)
        for regime in Regime:
            mask = labels.loc[common_idx] == regime
            count = mask.sum()
            regime_returns = returns.loc[common_idx][mask]

            stats[f"{regime.value}_count"] = count
            stats[f"{regime.value}_fraction"] = count / max(len(labels), 1)

            if count > 0 and regime_returns.dropna().size > 0:
                rets = regime_returns.dropna()
                stats[f"{regime.value}_mean_daily_return"] = float(rets.mean())
                stats[f"{regime.value}_vol_daily"] = float(rets.std())
                stats[f"{regime.value}_ann_return"] = float(
                    rets.mean() * _TRADING_DAYS_PER_YEAR
                )
                stats[f"{regime.value}_ann_vol"] = float(
                    rets.std() * np.sqrt(_TRADING_DAYS_PER_YEAR)
                )
            else:
                for key_suffix in (
                    "_mean_daily_return",
                    "_vol_daily",
                    "_ann_return",
                    "_ann_vol",
                ):
                    stats[f"{regime.value}{key_suffix}"] = np.nan

        return stats

    @staticmethod
    def _emit_coverage_warnings(
        regimes: dict[Regime, list[tuple[pd.Timestamp, pd.Timestamp]]],
        index: pd.DatetimeIndex,
    ) -> None:
        """Warn if any regime has insufficient or no coverage."""
        for regime in (
            Regime.HIGH_VOLATILITY_SHOCK,
            Regime.RISING_INTEREST_RATES,
            Regime.PROLONGED_SIDEWAYS,
        ):
            intervals = regimes.get(regime, [])
            total_obs = sum(
                len(index[(index >= start) & (index <= end)])
                for start, end in intervals
            )
            name = regime.value.replace("_", " ").title()
            if total_obs == 0:
                logger.warning(
                    "No %s regime detected in the data (%s to %s). "
                    "Regime invariance cannot be proven.",
                    name,
                    index[0].strftime("%Y-%m-%d"),
                    index[-1].strftime("%Y-%m-%d"),
                )
            elif total_obs < _MIN_OBSERVATIONS_PER_REGIME:
                logger.warning(
                    "Only %d observations found for %s regime (minimum recommended: %d). "
                    "Regime-specific statistics may be unreliable.",
                    total_obs,
                    name,
                    _MIN_OBSERVATIONS_PER_REGIME,
                )


# ---------------------------------------------------------------------------
# Purged K-Fold
# ---------------------------------------------------------------------------


class PurgedKFold:
    """Time-series cross-validation with purging and embargo.

    Standard k-fold cross-validation leaks information across adjacent folds
    because financial returns exhibit serial correlation. ``PurgedKFold``
    removes a buffer of observations between the training and test sets
    (purge) and enforces an additional gap (embargo) after each test set to
    prevent overlapping information.

    Parameters
    ----------
    n_splits:
        Number of folds.
    embargo_pct:
        Fraction of the total series length used as the embargo gap between
        the test set of fold ``i`` and the training set of fold ``i+1``.
        A value of ``0.01`` means 1 % of the total observations are dropped
        after each test set.
    purge_pct:
        Fraction of the series length used as the purge gap between the end
        of training and the start of the test set within each fold.
        Defaults to the same as *embargo_pct*.
    """

    def __init__(
        self,
        n_splits: int = _DEFAULT_N_SPLITS,
        *,
        embargo_pct: float = _DEFAULT_EMBARGO_PCT,
        purge_pct: float | None = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if not 0.0 <= embargo_pct < 0.5:
            raise ValueError("embargo_pct must be in [0.0, 0.5).")
        if purge_pct is not None and not 0.0 <= purge_pct < 0.5:
            raise ValueError("purge_pct must be in [0.0, 0.5).")

        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct if purge_pct is not None else embargo_pct

    def split(
        self,
        data: pd.DataFrame | pd.Series,
    ) -> list[PurgedFold]:
        """Generate purged train/test folds from a time-series object.

        Args:
            data: A ``pd.Series`` or ``pd.DataFrame`` with a ``DatetimeIndex``.

        Returns:
            A list of ``PurgedFold`` named tuples, one per split.

        Raises:
            TypeError: If *data* lacks a ``DatetimeIndex``.
            ValueError: If there are insufficient observations for the
                requested number of splits.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("data must have a DatetimeIndex.")

        n = len(data)
        min_needed = self.n_splits * 2 + 4
        if n < min_needed:
            raise ValueError(
                f"Need at least {min_needed} observations for "
                f"{self.n_splits} splits; got {n}."
            )

        embargo_size = int(n * self.embargo_pct) if self.embargo_pct > 0 else 0
        purge_size = int(n * self.purge_pct) if self.purge_pct > 0 else 0

        # Split into roughly equal-sized folds
        fold_size = (n - (self.n_splits - 1) * embargo_size) // self.n_splits
        if fold_size < 3:
            raise ValueError(
                f"Fold size ({fold_size}) is too small. Reduce n_splits or "
                f"increase the sample size."
            )

        timestamps = data.index
        folds: list[PurgedFold] = []

        for i in range(self.n_splits):
            test_start_idx = i * (fold_size + embargo_size)
            test_end_idx = test_start_idx + fold_size - 1

            if test_end_idx >= n:
                test_end_idx = n - 1
                test_start_idx = max(0, test_end_idx - fold_size + 1)

            # Training: everything before test, minus purge
            train_start_idx = 0
            train_end_idx = test_start_idx - purge_size - 1

            if train_end_idx < 0:
                # Not enough room for training before; use everything after
                train_start_idx = test_end_idx + embargo_size + 1
                train_end_idx = n - 1
                if train_start_idx >= n:
                    raise ValueError(f"Fold {i}: insufficient data for a training set.")

            purge_start = (
                timestamps[train_end_idx + 1]
                if train_end_idx + 1 < n and purge_size > 0
                else None
            )
            purge_end = (
                timestamps[test_start_idx - 1]
                if test_start_idx > 0 and purge_size > 0
                else None
            )

            folds.append(
                PurgedFold(
                    train_start=timestamps[train_start_idx],
                    train_end=timestamps[train_end_idx],
                    test_start=timestamps[test_start_idx],
                    test_end=timestamps[test_end_idx],
                    purge_start=purge_start,
                    purge_end=purge_end,
                )
            )

        return folds

    def split_indices(
        self,
        data: pd.DataFrame | pd.Series,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return integer-position index arrays for each train/test split.

        This is a convenience wrapper around ``split()`` that returns
        positional indices instead of timestamp bounds, making it
        compatible with ``sklearn``-style cross-validation loops.

        Args:
            data: A ``pd.Series`` or ``pd.DataFrame`` with a ``DatetimeIndex``.

        Returns:
            A list of ``(train_idx, test_idx)`` tuples as numpy arrays.
        """
        folds = self.split(data)
        timestamps = data.index
        result: list[tuple[np.ndarray, np.ndarray]] = []

        for fold in folds:
            train_mask = (timestamps >= fold.train_start) & (
                timestamps <= fold.train_end
            )
            test_mask = (timestamps >= fold.test_start) & (timestamps <= fold.test_end)
            result.append(
                (
                    np.where(train_mask)[0],
                    np.where(test_mask)[0],
                )
            )

        return result


# ---------------------------------------------------------------------------
# Stationary Block Bootstrap
# ---------------------------------------------------------------------------


def _stationary_block_bootstrap(
    returns: np.ndarray,
    block_length: int,
    n_blocks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a single bootstrap path using the stationary block bootstrap.

    In the stationary (Politis–Romano) bootstrap, block lengths are drawn
    from a geometric distribution with mean ``block_length``. Blocks are
    drawn with replacement from the original series, wrapping around
    circularly to handle end-of-series conditions.

    Args:
        returns: 1-D array of returns.
        block_length: Expected (mean) block length.
        n_blocks: Number of blocks to draw.
        rng: Seeded ``numpy.random.Generator``.

    Returns:
        1-D array of bootstrapped returns.
    """
    n = len(returns)
    if n < 2:
        return returns.copy()

    samples: list[float] = []
    for _ in range(n_blocks):
        # Geometric distribution: P(length=k) = p * (1-p)^(k-1)
        p = 1.0 / max(block_length, 1)
        blen = max(1, int(rng.geometric(p=p)))
        start = rng.integers(0, n)
        for offset in range(blen):
            idx = (start + offset) % n  # circular wrap
            samples.append(float(returns[idx]))

    return np.array(samples, dtype=float)


def bootstrap_regime_paths(
    returns: pd.Series,
    regime_labels: pd.Series,
    regime: Regime,
    *,
    block_length: int | None = None,
    n_paths: int = _DEFAULT_N_BOOTSTRAP,
    path_length: int | None = None,
    risk_free_rate: float = 0.0,
    random_state: int | None = None,
) -> BootstrapResult:
    """Generate synthetic equity curves via regime-clustered block bootstrap.

    Blocks are drawn **exclusively** from observations labelled with
    *regime*. Each bootstrap path preserves within-block autocorrelation
    through the stationary block bootstrap.

    Args:
        returns:
            Daily return series (decimal fractions) with a
            ``DatetimeIndex``.
        regime_labels:
            A ``Regime``-labelled series with the same index as *returns*.
        regime:
            The regime from which to draw blocks.
        block_length:
            Mean block length for the stationary bootstrap. When ``None``,
            the Politis–Romano optimal length is computed from the
            regime-specific returns.
        n_paths:
            Number of bootstrap paths to generate.
        path_length:
            Desired length of each synthetic path in observations. Defaults
            to the number of available regime observations.
        risk_free_rate:
            Annual risk-free rate (decimal) subtracted from returns before
            computing the Sharpe ratio.
        random_state:
            Seed for reproducibility.

    Returns:
        A ``BootstrapResult`` containing equity curves, Sharpe distribution,
        and bootstrap metadata.

    Raises:
        ValueError: If the regime has too few observations for bootstrapping.
    """
    rng = np.random.default_rng(random_state)

    # Extract regime-specific returns
    mask = regime_labels == regime
    regime_returns = returns.loc[returns.index.intersection(regime_labels[mask].index)]
    regime_returns = regime_returns.dropna()

    if len(regime_returns) < 2:
        raise ValueError(
            f"Regime {regime.value} has fewer than 2 observations; cannot bootstrap."
        )

    regime_arr = regime_returns.values

    if block_length is None:
        block_length = optimal_block_length(pd.Series(regime_arr))
    block_length = max(1, block_length)

    if path_length is None:
        path_length = len(regime_arr)
    path_length = max(1, path_length)

    # Number of blocks per path — draw extra blocks to ensure we have
    # enough observations after the random geometric-length draws.
    # Geometric distribution has heavy tail; 2x overshoot is safe.
    n_blocks_per_path = max(1, int(np.ceil(path_length * 2.0 / block_length)))

    daily_rf = risk_free_rate / _TRADING_DAYS_PER_YEAR

    equity_curves: list[pd.Series] = []
    sharpe_distribution = np.empty(n_paths, dtype=float)

    for i in range(n_paths):
        boot_returns = _stationary_block_bootstrap(
            regime_arr, block_length, n_blocks_per_path, rng
        )

        # Trim to exact path length
        boot_returns = boot_returns[:path_length]

        # Build equity curve
        equity = 100.0 * np.cumprod(1.0 + boot_returns)
        equity_curves.append(pd.Series(equity, name=f"bootstrap_{i}"))

        # Sharpe ratio
        excess = boot_returns - daily_rf
        mean_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1)

        if std_excess < 1e-12:
            sharpe_distribution[i] = 0.0
        else:
            sharpe_distribution[i] = float(
                (mean_excess / std_excess) * np.sqrt(_TRADING_DAYS_PER_YEAR)
            )

    return BootstrapResult(
        equity_curves=equity_curves,
        sharpe_distribution=sharpe_distribution,
        regime=regime,
        block_length=block_length,
        n_paths=n_paths,
    )


# ---------------------------------------------------------------------------
# Regime Survival Analysis
# ---------------------------------------------------------------------------


def compute_regime_survival_rates(
    returns: pd.Series,
    regime_labels: pd.Series,
    *,
    n_paths: int = _DEFAULT_N_BOOTSTRAP,
    risk_free_rate: float = 0.0,
    random_state: int | None = None,
) -> dict[Regime, float]:
    """Compute the fraction of bootstrap paths with positive Sharpe per regime.

    A survival rate of 1.0 means every resampled path from that regime
    produced a positive Sharpe ratio. Rates near 0.5 indicate chance-level
    performance. This is a non-parametric measure of strategy robustness
    within each structural regime.

    Args:
        returns: Daily return series.
        regime_labels: Regime-labelled series aligned with *returns*.
        n_paths: Bootstrap paths per regime.
        risk_free_rate: Annual risk-free rate.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary mapping each regime to its survival rate (float in [0, 1]).
    """
    rng = np.random.default_rng(random_state)
    survival: dict[Regime, float] = {}

    for regime in Regime:
        if regime == Regime.NORMAL:
            continue

        mask = regime_labels == regime
        if mask.sum() < 2:
            survival[regime] = float("nan")
            logger.warning(
                "Insufficient data for %s regime; survival rate not computed.",
                regime.value,
            )
            continue

        try:
            result = bootstrap_regime_paths(
                returns=returns,
                regime_labels=regime_labels,
                regime=regime,
                n_paths=n_paths,
                risk_free_rate=risk_free_rate,
                random_state=rng.integers(0, 2**31),
            )
            survival[regime] = float(np.mean(result.sharpe_distribution > 0))
        except ValueError:
            survival[regime] = float("nan")

    return survival


def check_regime_dependency(
    returns: pd.Series,
    regime_labels: pd.Series,
    *,
    n_paths: int = _DEFAULT_N_BOOTSTRAP,
    risk_free_rate: float = 0.0,
    dependency_threshold: float = 0.0,
    random_state: int | None = None,
) -> RegimeDependencyReport:
    """Diagnose whether a strategy's positive Sharpe depends on a single regime.

    Structural dependency is flagged when the strategy's Sharpe ratio is
    non-positive in one or more regimes but positive in at least one other.
    This indicates that the overall performance is not regime-invariant and
    may deteriorate when the macroeconomic environment shifts.

    Args:
        returns: Daily return series.
        regime_labels: Regime-labelled series.
        n_paths: Bootstrap paths per regime.
        risk_free_rate: Annual risk-free rate.
        dependency_threshold:
            Minimum mean Sharpe ratio in a regime to consider it supportive.
        random_state: Seed for reproducibility.

    Returns:
        A ``RegimeDependencyReport`` with survival rates and dependency flags.
    """
    survival = compute_regime_survival_rates(
        returns=returns,
        regime_labels=regime_labels,
        n_paths=n_paths,
        risk_free_rate=risk_free_rate,
        random_state=random_state,
    )

    # Identify regimes with data
    active_regimes = {r: s for r, s in survival.items() if not np.isnan(s)}

    if len(active_regimes) < 2:
        return RegimeDependencyReport(
            survival_rates=survival,
            structural_dependency=False,
            dependent_regime=None,
            warning_message=(
                "Fewer than two regimes have sufficient data; "
                "dependency analysis cannot be performed."
            ),
        )

    # Check for dependency: does Sharpe hold across all regimes?
    rates = list(active_regimes.values())
    min_rate = min(rates)
    max_rate = max(rates)

    # A regime is "dependent" if the strategy only works there
    dependent: bool = min_rate <= dependency_threshold < max_rate
    dependent_regime: Regime | None = None

    if dependent:
        # Find the regime(s) that are positive
        positive_regimes = [
            r for r, s in active_regimes.items() if s > dependency_threshold
        ]
        failing_regimes = [
            r for r, s in active_regimes.items() if s <= dependency_threshold
        ]

        if len(positive_regimes) == 1:
            dependent_regime = positive_regimes[0]

        failing_names = ", ".join(
            r.value.replace("_", " ").title() for r in failing_regimes
        )
        positive_names = ", ".join(
            r.value.replace("_", " ").title() for r in positive_regimes
        )

        message = (
            f"Structural dependency detected: positive Sharpe ratios are "
            f"observed in [{positive_names}] but fail in [{failing_names}]. "
            f"Strategy performance is not regime-invariant and may deteriorate "
            f"when macroeconomic conditions shift."
        )

        warnings.warn(message, RegimeDependencyWarning, stacklevel=2)

    else:
        message = None

    return RegimeDependencyReport(
        survival_rates=survival,
        structural_dependency=dependent,
        dependent_regime=dependent_regime,
        warning_message=message,
    )


__all__ = [
    "BootstrapResult",
    "PurgedFold",
    "PurgedKFold",
    "Regime",
    "RegimeDependencyReport",
    "RegimeDependencyWarning",
    "RegimeLabeler",
    "RegimeSegmentationResult",
    "bootstrap_regime_paths",
    "check_regime_dependency",
    "compute_regime_survival_rates",
    "optimal_block_length",
]
