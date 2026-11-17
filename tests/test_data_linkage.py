"""Deterministic unit tests for :class:`DataLinker.calculate_trend_signals`.

Uses synthetic price data with fixed values — no network calls.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from pysharpe.data.linkage import DataLinker, HistoryLinker


def _make_price_frame(
    prices: list[float],
    start_date: date | None = None,
) -> pd.DataFrame:
    """Build a ``(date, price)`` DataFrame from a list of prices."""
    start = start_date or date(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(len(prices))]
    return pd.DataFrame({"date": dates, "price": prices})


# ---------------------------------------------------------------------------
# Basic output structure
# ---------------------------------------------------------------------------


class TestCalculateTrendSignalsStructure:
    """Verify the output schema and edge-case handling."""

    EXPECTED_COLUMNS = [
        "date",
        "price",
        "price_rolling_avg",
        "price_lag_1",
        "short_ma",
        "long_ma",
        "ma_crossover_signal",
        "vol_30d",
        "vol_hist",
        "volatility_ratio",
    ]

    def test_returns_expected_columns(self):
        """All documented columns must be present."""
        linker = DataLinker()
        df = _make_price_frame([100.0, 101.0, 102.0, 103.0, 104.0])
        linker.register_data("market_data", df)
        result = linker.calculate_trend_signals(short_window=3, long_window=5)
        linker.close()
        assert list(result.columns) == self.EXPECTED_COLUMNS

    def test_insufficient_rows_returns_empty(self):
        """Fewer than 2 rows cannot produce log returns — return empty."""
        linker = DataLinker()
        df = _make_price_frame([100.0])
        linker.register_data("market_data", df)
        result = linker.calculate_trend_signals(short_window=3, long_window=5)
        linker.close()
        assert result.empty
        assert list(result.columns) == self.EXPECTED_COLUMNS

    def test_row_count_matches_input(self):
        """Output row count equals input row count (no rows dropped)."""
        linker = DataLinker()
        prices = [100.0 + i for i in range(30)]
        df = _make_price_frame(prices)
        linker.register_data("market_data", df)
        result = linker.calculate_trend_signals(short_window=5, long_window=10)
        linker.close()
        assert len(result) == len(prices)


# ---------------------------------------------------------------------------
# MA crossover signal (deterministic verification)
# ---------------------------------------------------------------------------


class TestMACrossoverSignal:
    """Verify the ``ma_crossover_signal`` column against manual computation."""

    # Prices that start flat, then rise, then fall:
    #    day 0-4:  increasing  100 → 104
    #    day 5-9:  decreasing  103 →  99
    #    day 10-14: increasing  100 → 104
    PRICES = [
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,  # 0-4
        103.0,
        102.0,
        101.0,
        100.0,
        99.0,  # 5-9
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,  # 10-14
    ]

    def test_signal_values_against_pandas_reference(self):
        """MA crossover signal must match a pure-pandas computation."""
        linker = DataLinker()
        df = _make_price_frame(self.PRICES)
        linker.register_data("market_data", df)

        short_w, long_w = 3, 5
        result = linker.calculate_trend_signals(
            short_window=short_w, long_window=long_w
        )
        linker.close()

        # --- Compute expected MA crossovers with pandas ---
        prices = pd.Series(self.PRICES, dtype=float)
        expected_short = prices.rolling(window=short_w, min_periods=1).mean()
        expected_long = prices.rolling(window=long_w, min_periods=1).mean()
        expected_signal = np.where(
            expected_short > expected_long,
            1,
            np.where(expected_short < expected_long, -1, 0),
        )

        for i in range(len(self.PRICES)):
            assert result["ma_crossover_signal"].iloc[i] == expected_signal[i], (
                f"Row {i}: expected {expected_signal[i]}, "
                f"got {result['ma_crossover_signal'].iloc[i]}"
            )

    def test_bullish_trend_gives_positive_signal(self):
        """Steadily rising prices should produce bullish (1) signals eventually."""
        linker = DataLinker()
        rising = [100.0 + i * 2 for i in range(20)]  # 100 → 138
        df = _make_price_frame(rising)
        linker.register_data("market_data", df)
        result = linker.calculate_trend_signals(short_window=5, long_window=10)
        linker.close()

        # After enough data, short MA should be above long MA
        last_signal = int(result["ma_crossover_signal"].iloc[-1])
        assert last_signal == 1, f"Expected bullish signal, got {last_signal}"

    def test_bearish_trend_gives_negative_signal(self):
        """Steadily falling prices should produce bearish (-1) signals eventually."""
        linker = DataLinker()
        falling = [200.0 - i * 2 for i in range(20)]  # 200 → 162
        df = _make_price_frame(falling)
        linker.register_data("market_data", df)
        result = linker.calculate_trend_signals(short_window=5, long_window=10)
        linker.close()

        last_signal = int(result["ma_crossover_signal"].iloc[-1])
        assert last_signal == -1, f"Expected bearish signal, got {last_signal}"

    def test_flat_prices_give_neutral_signal(self):
        """Identical prices should give 0 (neutral) once both MAs are populated."""
        linker = DataLinker()
        flat = [100.0] * 15
        df = _make_price_frame(flat)
        linker.register_data("market_data", df)
        result = linker.calculate_trend_signals(short_window=3, long_window=5)
        linker.close()

        # After row 4 (0-indexed), both short and long MAs should be 100.0
        assert all(result["ma_crossover_signal"].iloc[4:] == 0)


# ---------------------------------------------------------------------------
# Volatility ratio (deterministic verification)
# ---------------------------------------------------------------------------


class TestVolatilityRatio:
    """Verify the ``volatility_ratio`` column against manual computation."""

    @staticmethod
    def _annualised_vol(log_rets: np.ndarray, window: int | None = None) -> np.ndarray:
        """Compute annualised volatility from log returns.

        When *window* is ``None``, returns a **single scalar** — the
        full-period sample standard deviation.  This matches DuckDB's
        ``STDDEV_SAMP(...) OVER ()`` which returns the same value on
        every row.

        When *window* is an int, returns a rolling-window series matching
        DuckDB's ``ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW``.
        """
        s = pd.Series(log_rets)
        if window is None:
            return s.std(ddof=1) * np.sqrt(252)  # scalar
        return s.rolling(window=window, min_periods=2).std(ddof=1) * np.sqrt(252)

    def test_volatility_ratio_matches_manual_computation(self):
        """Volatility ratio must equal rolling-30d / full-period annualised vol."""
        # Use 31 days so the 30-day rolling window is fully populated at the end
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 1, 31))
        prices_list = [float(p) for p in prices]

        linker = DataLinker()
        df = _make_price_frame(prices_list)
        linker.register_data("market_data", df)
        result = linker.calculate_trend_signals(short_window=5, long_window=10)
        linker.close()

        # Manual computation
        log_rets = np.diff(np.log(prices))
        vol_30d_expected = self._annualised_vol(log_rets, window=30)
        vol_hist_expected = self._annualised_vol(log_rets, window=None)  # scalar

        # Compare each row
        for i in range(len(prices_list)):
            actual_rat = result["volatility_ratio"].iloc[i]

            if i < 1:
                # First row: no log return yet
                assert np.isnan(actual_rat), (
                    f"Row {i}: expected NaN (no log return), got {actual_rat}"
                )
            else:
                expected_vol_30d = vol_30d_expected.iloc[i - 1]
                if np.isnan(expected_vol_30d) or vol_hist_expected == 0:
                    assert np.isnan(actual_rat), (
                        f"Row {i}: expected NaN, got {actual_rat}"
                    )
                else:
                    expected_rat = float(expected_vol_30d) / float(vol_hist_expected)
                    assert actual_rat == pytest.approx(expected_rat, rel=1e-4), (
                        f"Row {i}: expected {expected_rat:.6f}, got {actual_rat:.6f}"
                    )

    def test_constant_prices_yield_nan_volatility_ratio(self):
        """Zero-variance prices → log returns are 0 → vol ratio is NaN."""
        linker = DataLinker()
        flat = [100.0] * 20
        df = _make_price_frame(flat)
        linker.register_data("market_data", df)
        result = linker.calculate_trend_signals(short_window=5, long_window=10)
        linker.close()

        # All log returns are 0, so vol_hist = 0 → ratio is NULL / NaN
        assert result["volatility_ratio"].isna().all()

    def test_high_volatility_ratio_when_recent_spike(self):
        """A price spike in the last 30 days should drive vol_ratio > 1."""
        # 100 days of calm, then 30 days of turbulence
        rng = np.random.default_rng(123)
        calm = 100.0 + np.cumsum(rng.normal(0, 0.1, 70))
        spike = calm[-1] + np.cumsum(rng.normal(0, 2.0, 30))
        prices_list = [float(p) for p in np.concatenate([calm, spike])]

        linker = DataLinker()
        df = _make_price_frame(prices_list)
        linker.register_data("market_data", df)
        result = linker.calculate_trend_signals(short_window=5, long_window=10)
        linker.close()

        last_ratio = float(result["volatility_ratio"].iloc[-1])
        assert last_ratio > 1.0, (
            f"Expected vol_ratio > 1 with recent spike, got {last_ratio:.4f}"
        )

    def test_low_volatility_ratio_when_recent_calm(self):
        """Calm recent period relative to turbulent history → vol_ratio < 1."""
        rng = np.random.default_rng(456)
        turbulent = 100.0 + np.cumsum(rng.normal(0, 2.0, 70))
        calm = turbulent[-1] + np.cumsum(rng.normal(0, 0.1, 30))
        prices_list = [float(p) for p in np.concatenate([turbulent, calm])]

        linker = DataLinker()
        df = _make_price_frame(prices_list)
        linker.register_data("market_data", df)
        result = linker.calculate_trend_signals(short_window=5, long_window=10)
        linker.close()

        last_ratio = float(result["volatility_ratio"].iloc[-1])
        # Should be well below 1 since recent period is much calmer
        assert last_ratio < 0.5, (
            f"Expected vol_ratio < 0.5 with calm recent period, got {last_ratio:.4f}"
        )


# ---------------------------------------------------------------------------
# HistoryLinker: stitched proxy history (restored coverage + FX guardrail)
# ---------------------------------------------------------------------------


class _HistoryFetcher:
    """Deterministic in-memory fetcher for :class:`HistoryLinker` tests.

    Data (UTC business days):
    - TARGET: exists from 2020-01-05 onwards (100.0 → 105.0).
    - PROXY:  exists for all 10 days (10.0 → 19.0).
    - USDCAD=X: flat 2.0 for all 10 days.
    """

    def fetch_history(
        self,
        ticker: str,
        *,
        period: str = "max",
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        if ticker == "TARGET":
            return pd.DataFrame(
                {"Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
                index=dates[4:],
            )
        if ticker == "PROXY":
            return pd.DataFrame(
                {
                    "Close": [
                        10.0,
                        11.0,
                        12.0,
                        13.0,
                        14.0,
                        15.0,
                        16.0,
                        17.0,
                        18.0,
                        19.0,
                    ]
                },
                index=dates,
            )
        if ticker == "USDCAD=X":
            return pd.DataFrame({"Close": [2.0] * 10}, index=dates)
        return pd.DataFrame()


class TestHistoryLinkerStitched:
    """Proxy stitching: handover math, FX adjustment, and coverage guardrails."""

    @staticmethod
    def _target_series() -> pd.Series:
        dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        return pd.Series(
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            index=dates[4:],
            name="TARGET",
        )

    def test_stitched_series_no_fx(self):
        """Without FX adjustment the proxy is spliced at the handover date."""
        linker = HistoryLinker(
            proxy_map={"TARGET": "PROXY"},
            fx_adjust=False,
            fetcher=_HistoryFetcher(),
        )
        stitched = linker.get_stitched_series("TARGET", start_date="2020-01-01")

        # Handover 2020-01-05: target 100.0, proxy 14.0 → scalar 100/14
        assert len(stitched) == 10
        assert stitched.name == "TARGET"
        assert stitched.iloc[4] == pytest.approx(100.0)
        assert stitched.iloc[-1] == pytest.approx(105.0)
        assert stitched.iloc[0] == pytest.approx(10.0 * (100.0 / 14.0))

    def test_stitched_series_with_fx(self):
        """FX adjustment prices the proxy portion at USD * USDCAD."""
        linker = HistoryLinker(
            proxy_map={"TARGET": "PROXY"},
            fx_adjust=True,
            fetcher=_HistoryFetcher(),
        )
        stitched = linker.get_stitched_series("TARGET", start_date="2020-01-01")

        # FX-adjusted proxy at T0: 14.0 * 2.0 = 28.0 -> scalar 100/28
        assert len(stitched) == 10
        assert stitched.iloc[4] == pytest.approx(100.0)
        assert stitched.iloc[-1] == pytest.approx(105.0)
        assert stitched.iloc[0] == pytest.approx((10.0 * 2.0) * (100.0 / 28.0))

    def test_stitched_series_fx_direction_with_varying_usdcad(self):
        """When USDCAD=X rises at constant USD proxy price, CAD proxy rises."""

        class _VaryingFxFetcher(_HistoryFetcher):
            def fetch_history(self, ticker, **kwargs):
                dates = pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC")
                if ticker == "PROXY":
                    return pd.DataFrame({"Close": [10.0] * 6}, index=dates)
                if ticker == "TARGET":
                    return pd.DataFrame({"Close": [100.0, 101.0]}, index=dates[4:])
                if ticker == "USDCAD=X":
                    # USD strengthens from 1.0 CAD to 2.0 CAD before handover (Jan 5)
                    return pd.DataFrame(
                        {"Close": [1.0, 1.0, 1.5, 1.5, 2.0, 2.0]},
                        index=dates,
                    )
                return super().fetch_history(ticker, **kwargs)

        linker = HistoryLinker(
            proxy_map={"TARGET": "PROXY"},
            fx_adjust=True,
            fetcher=_VaryingFxFetcher(),
        )
        stitched = linker.get_stitched_series("TARGET", start_date="2020-01-01")

        # At handover (Jan 5): proxy CAD = 10 * 2.0 = 20 -> scalar = 100/20 = 5.
        # On Jan 1 (USDCAD=1.0 vs 2.0 at handover): CAD equivalent is $50 (not $200).
        assert stitched.loc[pd.Timestamp("2020-01-01", tz="UTC")] == pytest.approx(50.0)
        assert stitched.loc[pd.Timestamp("2020-01-03", tz="UTC")] == pytest.approx(75.0)
        assert stitched.loc[pd.Timestamp("2020-01-05", tz="UTC")] == pytest.approx(
            100.0
        )

    def test_stitched_series_no_proxy_defined(self):
        """A target without a proxy mapping returns its raw series."""
        linker = HistoryLinker(
            proxy_map={"OTHER": "PROXY"},
            fx_adjust=False,
            fetcher=_HistoryFetcher(),
        )
        stitched = linker.get_stitched_series("TARGET", start_date="2020-01-01")

        assert len(stitched) == 6
        assert stitched.index.min() == pd.Timestamp("2020-01-05", tz="UTC")

    def test_stitched_series_fx_gap_excludes_uncovered_rows(self):
        """Proxy rows before the first FX rate must be excluded, never
        backfilled with a future exchange rate (lookahead bias)."""

        class _GappedFxFetcher(_HistoryFetcher):
            def fetch_history(
                self,
                ticker: str,
                *,
                period: str = "max",
                interval: str = "1d",
                start: str | None = None,
                end: str | None = None,
            ) -> pd.DataFrame:
                frame = super().fetch_history(
                    ticker, period=period, interval=interval, start=start, end=end
                )
                if ticker == "USDCAD=X":
                    # Rate is only known from 2020-01-05 onwards and changes
                    # afterwards (2.0 → 3.0), so a backfill is observable.
                    return pd.DataFrame(
                        {"Close": [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]},
                        index=frame.index[4:],
                    )
                return frame

        linker = HistoryLinker(
            proxy_map={"TARGET": "PROXY"},
            fx_adjust=True,
            fetcher=_GappedFxFetcher(),
        )
        stitched = linker.get_stitched_series("TARGET", start_date="2020-01-01")

        # The four proxy rows before 2020-01-05 have no rate coverage: the
        # stitched series must start at the first FX-covered date instead of
        # being priced with the 2020-01-05 rate retroactively.
        assert stitched.index.min() == pd.Timestamp("2020-01-05", tz="UTC")
        assert len(stitched) == 6
        assert stitched.notna().all()

    def test_stitched_series_unsorted_proxy_preserves_causal_fx_prefix(self):
        class _UnsortedProxyFetcher(_HistoryFetcher):
            def __init__(self, include_future: bool):
                self.include_future = include_future

            def fetch_history(self, ticker, **kwargs):
                frame = super().fetch_history(ticker, **kwargs)
                if ticker == "PROXY":
                    # Keep the handover on Jan 5 in both histories, but place
                    # the later FX observation before the earlier proxy rows.
                    positions = (
                        [7, 0, 2, 1, 3, 4, 5, 6]
                        if self.include_future
                        else list(range(7))
                    )
                    return frame.iloc[positions]
                if ticker == "USDCAD=X":
                    if self.include_future:
                        return pd.DataFrame(
                            {"Close": [2.0, 3.0]}, index=frame.index[[1, 7]]
                        )
                    return pd.DataFrame({"Close": [2.0]}, index=frame.index[[1]])
                return frame

        def stitch(include_future: bool) -> pd.Series:
            return HistoryLinker(
                proxy_map={"TARGET": "PROXY"},
                fx_adjust=True,
                fetcher=_UnsortedProxyFetcher(include_future),
            ).get_stitched_series("TARGET", start_date="2020-01-01")

        prefix = stitch(False)
        full = stitch(True)
        cutoff = pd.Timestamp("2020-01-07", tz="UTC")
        assert pd.Timestamp("2020-01-01", tz="UTC") not in full.index
        pd.testing.assert_series_equal(
            full.loc[:cutoff], prefix.loc[:cutoff], check_freq=False
        )
        assert full.loc[pd.Timestamp("2020-01-03", tz="UTC")] == pytest.approx(
            12.0 * (100.0 / 14.0)
        )

    def test_stitched_series_fx_no_overlap_returns_target(self):
        """Zero FX overlap degrades gracefully to the raw target series."""

        class _NoFxFetcher(_HistoryFetcher):
            def fetch_history(
                self,
                ticker: str,
                *,
                period: str = "max",
                interval: str = "1d",
                start: str | None = None,
                end: str | None = None,
            ) -> pd.DataFrame:
                frame = super().fetch_history(
                    ticker, period=period, interval=interval, start=start, end=end
                )
                if ticker == "USDCAD=X":
                    # Rate window does not overlap the proxy window at all.
                    return pd.DataFrame(
                        {"Close": [2.0]},
                        index=[pd.Timestamp("2019-12-01", tz="UTC")],
                    )
                return frame

        linker = HistoryLinker(
            proxy_map={"TARGET": "PROXY"},
            fx_adjust=True,
            fetcher=_NoFxFetcher(),
        )
        stitched = linker.get_stitched_series("TARGET", start_date="2020-01-01")

        expected = self._target_series()
        assert len(stitched) == 6
        assert stitched.index.equals(expected.index)
        assert stitched.values == pytest.approx(expected.values)

    def test_stitched_series_unsorted_24_7_calendar_does_not_lookahead(self):
        """Unsorted 24/7 crypto proxy or SPY calendar must sort causally before ffill."""

        class _CryptoFetcher(_HistoryFetcher):
            def fetch_history(self, ticker, **kwargs):
                dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
                if ticker == "BTC-USD":
                    # Provide unordered proxy observations on odd days + handover day
                    idx = dates[[5, 1, 3, 4]]
                    return pd.DataFrame({"Close": [50.0, 10.0, 30.0, 40.0]}, index=idx)
                if ticker == "SPY":
                    # Unsorted equity calendar including Jan 2 and Jan 4
                    return pd.DataFrame(
                        {"Close": [1.0, 1.0, 1.0]},
                        index=dates[[4, 2, 0]],
                    )
                return super().fetch_history(ticker, **kwargs)

        linker = HistoryLinker(
            proxy_map={"TARGET": "BTC-USD"},
            fx_adjust=False,
            fetcher=_CryptoFetcher(),
        )
        stitched = linker.get_stitched_series("TARGET", start_date="2020-01-01")

        # Jan 2 is causally filled with Jan 1 proxy (10.0), not Jan 3/5 future values;
        # Jan 0 has no prior proxy observation and is excluded cleanly.
        assert pd.Timestamp("2020-01-01", tz="UTC") not in stitched.index
        assert stitched.loc[pd.Timestamp("2020-01-03", tz="UTC")] == pytest.approx(
            10.0 * (100.0 / 40.0)
        )

    def test_stitched_series_excludes_stale_fx_rows_beyond_limit(self):
        """HistoryLinker excludes proxy dates whose carried FX exceeds max_fx_staleness_days."""
        with pytest.raises(ValueError, match="non-negative"):
            HistoryLinker(proxy_map={"TARGET": "PROXY"}, max_fx_staleness_days=-1)

        class _StaleFxFetcher(_HistoryFetcher):
            def fetch_history(
                self,
                ticker: str,
                *,
                period: str = "max",
                interval: str = "1d",
                start: str | None = None,
                end: str | None = None,
            ) -> pd.DataFrame:
                # Extend proxy/FX dates across a >7-day FX blackout before handover.
                all_dates = pd.date_range("2020-01-01", periods=15, freq="D", tz="UTC")
                if ticker == "PROXY":
                    return pd.DataFrame(
                        {"Close": [float(10 + i) for i in range(15)]}, index=all_dates
                    )
                if ticker == "TARGET":
                    return pd.DataFrame(
                        {"Close": [100.0, 101.0, 102.0]}, index=all_dates[12:]
                    )
                if ticker == "USDCAD=X":
                    # Quoted on Jan 1 and Jan 12: Jan 9..11 (+8d..+10d) exceed 7d limit.
                    return pd.DataFrame(
                        {"Close": [1.5, 2.0]},
                        index=all_dates[[0, 11]],
                    )
                return super().fetch_history(
                    ticker, period=period, interval=interval, start=start, end=end
                )

        bounded = HistoryLinker(
            proxy_map={"TARGET": "PROXY"},
            fx_adjust=True,
            fetcher=_StaleFxFetcher(),
        ).get_stitched_series("TARGET", start_date="2020-01-01")

        # Jan 9..11 (indices 8, 9, 10) must be excluded; Jan 8 (+7d) and Jan 12 (+0d) kept.
        for stale_day in ["2020-01-09", "2020-01-10", "2020-01-11"]:
            assert pd.Timestamp(stale_day, tz="UTC") not in bounded.index
        assert pd.Timestamp("2020-01-08", tz="UTC") in bounded.index
        assert pd.Timestamp("2020-01-12", tz="UTC") in bounded.index

        unbounded = HistoryLinker(
            proxy_map={"TARGET": "PROXY"},
            fx_adjust=True,
            fetcher=_StaleFxFetcher(),
            max_fx_staleness_days=None,
        ).get_stitched_series("TARGET", start_date="2020-01-01")
        assert pd.Timestamp("2020-01-09", tz="UTC") in unbounded.index


# ---------------------------------------------------------------------------
# Window parameter propagation
# ---------------------------------------------------------------------------


class TestWindowParameters:
    """Verify that ``short_window`` and ``long_window`` are correctly forwarded
    to :meth:`get_enhanced_market_data`."""

    def test_custom_windows_produce_different_signals(self):
        """Different window sizes should produce different MA crossover results."""
        prices = [100.0, 102.0, 101.0, 103.0, 100.0, 102.0, 101.0, 104.0, 103.0, 105.0]

        linker_a = DataLinker()
        df_a = _make_price_frame(prices)
        linker_a.register_data("market_data", df_a)
        result_a = linker_a.calculate_trend_signals(short_window=2, long_window=8)
        linker_a.close()

        linker_b = DataLinker()
        df_b = _make_price_frame(prices)
        linker_b.register_data("market_data", df_b)
        result_b = linker_b.calculate_trend_signals(short_window=8, long_window=2)
        linker_b.close()

        # The signals must differ because short/long are swapped
        assert not result_a["ma_crossover_signal"].equals(
            result_b["ma_crossover_signal"]
        ), "Swapping short/long windows should produce different crossover signals"
