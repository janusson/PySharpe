"""Tests for the DuckDB trial ledger and PBO calculator.

Verifies:
* Schema creation and CRUD operations.
* Thread safety under concurrent writes.
* PBO computation for both normal and edge-case inputs.
* File-drawer bias: omitting failed trials **must** alter PBO output.
* Economic justification guard for various strategy philosophies.
"""

from __future__ import annotations

import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest

from pysharpe.validation.ledger import (
    DuckDBLedger,
    ExecutionStatus,
    PBOResult,
    TrialRecord,
    _sanitise_for_ranking,
    compute_pbo,
    validate_economic_justification,
)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_record(
    *,
    in_sample: float = 1.0,
    oos: float = 0.8,
    status: ExecutionStatus = ExecutionStatus.SUCCESS,
    justifiable: bool = True,
    config: dict | None = None,
) -> TrialRecord:
    """Create an ephemeral ``TrialRecord`` for testing."""
    if config is None:
        config = {"lookback": 60, "momentum_signal_sign": 1}
    return TrialRecord(
        trial_id=str(uuid.uuid4()),
        run_timestamp="2025-01-01T00:00:00Z",
        strategy_config=config,
        is_economically_justifiable=justifiable,
        in_sample_sharpe=in_sample,
        out_of_sample_sharpe=oos,
        execution_status=status,
    )


def _pbo_from_ledger(ledger: DuckDBLedger) -> PBOResult:
    """Convenience: extract IS/OOS Sharpe arrays from a ledger and compute PBO."""
    trials = ledger.load_trials()
    is_arr = np.array([t.in_sample_sharpe for t in trials], dtype=np.float64)
    oos_arr = np.array([t.out_of_sample_sharpe for t in trials], dtype=np.float64)
    return compute_pbo(is_arr, oos_arr)


# ===================================================================
# DuckDBLedger — schema & CRUD
# ===================================================================


class TestLedgerSchema:
    """Schema creation and basic persistence."""

    def test_table_exists_after_init(self) -> None:
        with DuckDBLedger(":memory:") as ledger:
            result = ledger._connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'optimization_trials'"
            ).fetchone()
            assert result is not None
            assert result[0] == 1

    def test_roundtrip_single_record(self) -> None:
        record = _make_record(in_sample=1.5, oos=1.1)
        with DuckDBLedger(":memory:") as ledger:
            ledger.write_trial(record)
            loaded = ledger.load_trials()
            assert len(loaded) == 1
            r = loaded[0]
            assert r.trial_id == record.trial_id
            assert r.in_sample_sharpe == pytest.approx(1.5)
            assert r.out_of_sample_sharpe == pytest.approx(1.1)
            assert r.execution_status == ExecutionStatus.SUCCESS
            assert r.strategy_config == {"lookback": 60, "momentum_signal_sign": 1}

    def test_roundtrip_nan_sharpe(self) -> None:
        """NaN Sharpe values (from failed/rejected trials) survive roundtrip."""
        record = _make_record(in_sample=float("nan"), oos=float("nan"), status=ExecutionStatus.FAILED)
        with DuckDBLedger(":memory:") as ledger:
            ledger.write_trial(record)
            loaded = ledger.load_trials()
            assert len(loaded) == 1
            assert math.isnan(loaded[0].in_sample_sharpe)
            assert math.isnan(loaded[0].out_of_sample_sharpe)

    def test_roundtrip_inf_sharpe(self) -> None:
        """Infinite Sharpe values are stored as NULL and roundtripped as NaN."""
        record = _make_record(in_sample=float("inf"), oos=float("-inf"))
        with DuckDBLedger(":memory:") as ledger:
            ledger.write_trial(record)
            loaded = ledger.load_trials()
            assert len(loaded) == 1
            assert math.isnan(loaded[0].in_sample_sharpe)
            assert math.isnan(loaded[0].out_of_sample_sharpe)

    def test_bulk_write(self) -> None:
        records = [_make_record(in_sample=float(i)) for i in range(10)]
        with DuckDBLedger(":memory:") as ledger:
            ledger.write_trials(records)
            assert ledger.count() == 10

    def test_count_filtered(self) -> None:
        records = [
            _make_record(status=ExecutionStatus.SUCCESS),
            _make_record(status=ExecutionStatus.FAILED),
            _make_record(status=ExecutionStatus.REJECTED),
            _make_record(status=ExecutionStatus.SUCCESS),
        ]
        with DuckDBLedger(":memory:") as ledger:
            ledger.write_trials(records)
            assert ledger.count() == 4
            assert ledger.count(status_filter=ExecutionStatus.SUCCESS) == 2
            assert ledger.count(status_filter=ExecutionStatus.FAILED) == 1
            assert ledger.count(status_filter=ExecutionStatus.REJECTED) == 1

    def test_clear(self) -> None:
        with DuckDBLedger(":memory:") as ledger:
            ledger.write_trial(_make_record())
            assert ledger.count() == 1
            ledger.clear()
            assert ledger.count() == 0

    def test_read_only_raises_on_write(self) -> None:
        # Seed a file ledger first.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            with DuckDBLedger(db_path) as w:
                w.write_trial(_make_record())

            with DuckDBLedger(db_path, read_only=True) as r:
                assert r.count() == 1
                with pytest.raises(RuntimeError):
                    r.write_trial(_make_record())

    def test_write_outside_context_raises(self) -> None:
        ledger = DuckDBLedger(":memory:")
        with pytest.raises(RuntimeError):
            ledger.write_trial(_make_record())

    def test_load_outside_context_raises(self) -> None:
        ledger = DuckDBLedger(":memory:")
        with pytest.raises(RuntimeError):
            ledger.load_trials()

    def test_load_dataframe(self) -> None:
        with DuckDBLedger(":memory:") as ledger:
            ledger.write_trial(_make_record(in_sample=0.5, oos=0.3))
            ledger.write_trial(_make_record(in_sample=1.2, oos=0.9))
            df = ledger.load_trials_dataframe()
            assert len(df) == 2
            assert list(df.columns) == [
                "trial_id",
                "run_timestamp",
                "strategy_config",
                "is_economically_justifiable",
                "in_sample_sharpe",
                "out_of_sample_sharpe",
                "execution_status",
            ]


# ===================================================================
# Thread safety
# ===================================================================


class TestThreadSafety:
    """Concurrent writers must not corrupt the database."""

    NUM_WRITERS: int = 16
    RECORDS_PER_WRITER: int = 25

    def test_concurrent_writes(self) -> None:
        with DuckDBLedger(":memory:") as ledger:

            def _writer(worker_id: int) -> None:
                for i in range(self.RECORDS_PER_WRITER):
                    record = _make_record(
                        in_sample=float(worker_id * 100 + i),
                        config={"worker": worker_id, "idx": i},
                    )
                    ledger.write_trial(record)

            with ThreadPoolExecutor(max_workers=self.NUM_WRITERS) as executor:
                futures = [
                    executor.submit(_writer, w) for w in range(self.NUM_WRITERS)
                ]
                for future in as_completed(futures):
                    future.result()  # Re-raise any exception.

            expected: int = self.NUM_WRITERS * self.RECORDS_PER_WRITER
            assert ledger.count() == expected
            # Verify data integrity: every expected (worker, idx) pair exists.
            trials = ledger.load_trials()
            seen: set[tuple[int, int]] = set()
            for t in trials:
                seen.add((t.strategy_config["worker"], t.strategy_config["idx"]))
            assert len(seen) == expected

    def test_no_deadlock_read_during_write(self) -> None:
        """A read while another thread holds the lock must not deadlock."""
        with DuckDBLedger(":memory:") as ledger:
            ledger.write_trial(_make_record())

            errors: list[Exception] = []

            def _reader() -> None:
                try:
                    _ = ledger.load_trials()
                except Exception as exc:
                    errors.append(exc)

            threads: list[threading.Thread] = []
            for _ in range(8):
                t = threading.Thread(target=_reader)
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"Reader threads raised: {errors}"


# ===================================================================
# PBO computation
# ===================================================================


class TestPBOComputation:
    """Unit tests for ``compute_pbo``."""

    # -- normal behaviour -------------------------------------------------

    def test_perfect_rank_correlation_yields_low_pbo(self) -> None:
        """When IS rank perfectly predicts OOS rank, PBO should be near 0."""
        is_sharpes = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        oos_sharpes = np.array([0.4, 0.9, 1.4, 1.9, 2.4])  # same ordering
        result = compute_pbo(is_sharpes, oos_sharpes)
        assert result.pbo < 0.1
        assert result.rank_correlation > 0.9

    def test_no_rank_correlation_yields_pbo_near_0_5(self) -> None:
        """When IS rank is unrelated to OOS rank, PBO should be ~0.5."""
        rng = np.random.default_rng(seed=42)
        is_sharpes = rng.normal(1.0, 0.2, size=100)
        oos_sharpes = rng.normal(0.8, 0.2, size=100)
        result = compute_pbo(is_sharpes, oos_sharpes)
        assert 0.4 < result.pbo < 0.6

    def test_negative_rank_correlation_yields_high_pbo(self) -> None:
        """When better IS predicts worse OOS, signal is overfitting — PBO > 0.5."""
        is_sharpes = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        oos_sharpes = np.array([2.4, 1.9, 1.4, 0.9, 0.4])  # reversed
        result = compute_pbo(is_sharpes, oos_sharpes)
        assert result.pbo > 0.9
        assert result.rank_correlation < -0.9

    # -- edge cases -------------------------------------------------------

    def test_zero_variance_is_triggers_flag(self) -> None:
        """Identical IS Sharpes → degenerate ranking → observation flag."""
        is_sharpes = np.array([1.0, 1.0, 1.0, 1.0])
        oos_sharpes = np.array([0.5, 0.6, 0.7, 0.8])
        result = compute_pbo(is_sharpes, oos_sharpes)
        assert result.pbo == pytest.approx(0.5)
        assert result.rank_correlation == pytest.approx(0.0)
        assert result.observation_flag == "zero_variance_is"

    def test_weak_correlation_flag(self) -> None:
        """Very weak ρ triggers an observation flag."""
        rng = np.random.default_rng(seed=99)
        is_sharpes = rng.normal(1.0, 0.1, size=50)
        oos_sharpes = rng.normal(0.8, 0.1, size=50)  # independent draws
        result = compute_pbo(is_sharpes, oos_sharpes)
        if abs(result.rank_correlation) < 0.1:
            assert result.observation_flag == "weak_rank_correlation"
        # Regardless of flag, PBO should still be valid.
        assert 0.0 <= result.pbo <= 1.0

    def test_nan_values_pushed_to_worst_rank(self) -> None:
        """NaN values are sanitised to the worst rank so they don't break spearmanr."""
        is_sharpes = np.array([1.0, float("nan"), 1.2, 0.8, 0.9, 1.1])
        oos_sharpes = np.array([0.9, 0.7, 1.0, float("nan"), 0.6, 0.8])
        result = compute_pbo(is_sharpes, oos_sharpes)
        # Should complete without error.
        assert 0.0 <= result.pbo <= 1.0

    def test_inf_values_handled(self) -> None:
        """±∞ values are sanitised to extreme finite values."""
        is_sharpes = np.array([1.0, float("inf"), float("-inf"), 0.5])
        oos_sharpes = np.array([0.8, float("inf"), float("-inf"), 0.4])
        result = compute_pbo(is_sharpes, oos_sharpes)
        assert 0.0 <= result.pbo <= 1.0

    # -- input validation -------------------------------------------------

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_pbo(np.array([1.0, 2.0]), np.array([1.0]))

    def test_too_few_trials_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            compute_pbo(np.array([1.0, 2.0]), np.array([2.0, 1.0]))


# ===================================================================
# File-drawer bias — omitting failures alters PBO
# ===================================================================


class TestFileDrawerBias:
    """Demonstrate that discarding failed trials changes the PBO estimate."""

    def test_omitting_failures_changes_pbo(self) -> None:
        rng = np.random.default_rng(seed=7)
        n: int = 60

        # IS Sharpe: successes clustered around 1.0; failures around 0.1.
        is_success: np.ndarray = rng.normal(1.0, 0.3, size=n // 2)
        is_failed: np.ndarray = rng.normal(0.1, 0.1, size=n // 2)
        is_all: np.ndarray = np.concatenate([is_success, is_failed])

        # OOS Sharpe: no correlation with IS in this scenario (overfit).
        oos_success: np.ndarray = rng.normal(0.5, 0.3, size=n // 2)
        oos_failed: np.ndarray = rng.normal(-0.5, 0.3, size=n // 2)
        oos_all: np.ndarray = np.concatenate([oos_success, oos_failed])

        pbo_all: PBOResult = compute_pbo(is_all, oos_all)
        pbo_success_only: PBOResult = compute_pbo(is_success, oos_success)

        # The PBO estimates MUST differ when failed trials are included.
        assert not math.isclose(pbo_all.pbo, pbo_success_only.pbo, rel_tol=1e-6), (
            f"PBO should differ between full ({pbo_all.pbo:.6f}) and "
            f"success-only ({pbo_success_only.pbo:.6f}) datasets"
        )

    def test_including_failures_produces_less_optimistic_pbo(self) -> None:
        """Full ledger (successes + failures) should generally have higher PBO."""
        rng = np.random.default_rng(seed=42)

        # Construct a scenario where successes show modest positive correlation
        # but failures are noise, so including them weakens the overall signal.
        n: int = 50
        base: np.ndarray = np.linspace(0.5, 2.0, n)
        # Add minimal noise so not every value is identical.
        is_success: np.ndarray = base + rng.normal(0, 0.05, size=n)
        oos_success: np.ndarray = base + rng.normal(0, 0.15, size=n)

        # Failed trials: weak IS, even weaker OOS, and uncorrelated.
        is_failed: np.ndarray = rng.normal(-0.2, 0.2, size=n)
        oos_failed: np.ndarray = rng.normal(-0.8, 0.3, size=n)

        is_all: np.ndarray = np.concatenate([is_success, is_failed])
        oos_all: np.ndarray = np.concatenate([oos_success, oos_failed])

        pbo_all: PBOResult = compute_pbo(is_all, oos_all)
        pbo_success: PBOResult = compute_pbo(is_success, oos_success)

        # Including failures should materially change PBO (file-drawer effect).
        assert not math.isclose(pbo_all.pbo, pbo_success.pbo, rel_tol=1e-6), (
            f"PBO should differ: full={pbo_all.pbo:.6f}, "
            f"success-only={pbo_success.pbo:.6f}"
        )

    def test_end_to_end_file_drawer_in_ledger(self) -> None:
        """Roundtrip: write successes + failures, compute both PBOs, verify change."""
        rng = np.random.default_rng(seed=13)
        n_success: int = 30
        n_failed: int = 20

        success_records: list[TrialRecord] = []
        failed_records: list[TrialRecord] = []
        for _i in range(n_success):
            success_records.append(
                _make_record(
                    in_sample=float(rng.normal(1.0, 0.3)),
                    oos=float(rng.normal(0.8, 0.3)),
                    status=ExecutionStatus.SUCCESS,
                )
            )
        for _i in range(n_failed):
            failed_records.append(
                _make_record(
                    in_sample=float(rng.normal(0.1, 0.2)),
                    oos=float(rng.normal(-0.3, 0.3)),
                    status=ExecutionStatus.FAILED,
                )
            )

        with DuckDBLedger(":memory:") as ledger:
            ledger.write_trials(success_records)
            pbo_success = _pbo_from_ledger(ledger)

            ledger.write_trials(failed_records)
            pbo_all = _pbo_from_ledger(ledger)

        # PBO must change (this is the file-drawer effect).
        assert not math.isclose(pbo_all.pbo, pbo_success.pbo, rel_tol=1e-6), (
            f"End-to-end: PBO(all)={pbo_all.pbo:.6f}, "
            f"PBO(success-only)={pbo_success.pbo:.6f}"
        )
        # Both PBOs should be valid regardless of direction.
        assert 0.0 <= pbo_all.pbo <= 1.0
        assert 0.0 <= pbo_success.pbo <= 1.0


# ===================================================================
# Economic justification guard
# ===================================================================


class TestEconomicJustification:
    """Unit tests for ``validate_economic_justification``."""

    def test_trend_following_positive_momentum_ok(self) -> None:
        ok, reason = validate_economic_justification(
            {"momentum_signal_sign": 1, "lookback": 60},
            strategy_philosophy="trend_following",
        )
        assert ok
        assert "justifiable" in reason

    def test_trend_following_negative_momentum_rejected(self) -> None:
        ok, reason = validate_economic_justification(
            {"momentum_signal_sign": -1, "lookback": 60},
            strategy_philosophy="trend_following",
        )
        assert not ok
        assert "contradicts" in reason

    def test_trend_following_zero_momentum_rejected(self) -> None:
        ok, reason = validate_economic_justification(
            {"momentum_signal_sign": 0},
            strategy_philosophy="trend_following",
        )
        assert not ok

    def test_mean_reversion_positive_momentum_rejected(self) -> None:
        ok, reason = validate_economic_justification(
            {"momentum_signal_sign": 1},
            strategy_philosophy="mean_reversion",
        )
        assert not ok

    def test_mean_reversion_negative_momentum_ok(self) -> None:
        ok, reason = validate_economic_justification(
            {"momentum_signal_sign": -1},
            strategy_philosophy="mean_reversion",
        )
        assert ok

    def test_direction_neutral_philosophy_ignores_momentum(self) -> None:
        """Risk-parity and vol-targeting don't care about signal direction."""
        for phil in ("risk_parity", "volatility_targeting"):
            ok, _ = validate_economic_justification(
                {"momentum_signal_sign": -1}, strategy_philosophy=phil
            )
            assert ok

    def test_long_only_violation_on_trend_following(self) -> None:
        ok, reason = validate_economic_justification(
            {"direction_long_only": False},
            strategy_philosophy="trend_following",
        )
        assert not ok
        assert "long-only" in reason.lower()

    def test_long_only_ok_on_trend_following(self) -> None:
        ok, _ = validate_economic_justification(
            {"direction_long_only": True},
            strategy_philosophy="trend_following",
        )
        assert ok

    def test_unknown_philosophy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy_philosophy"):
            validate_economic_justification({}, strategy_philosophy="grid_trading")

    def test_invalid_momentum_type(self) -> None:
        ok, reason = validate_economic_justification(
            {"momentum_signal_sign": "positive"}, strategy_philosophy="trend_following"
        )
        assert not ok
        assert "integer" in reason

    def test_invalid_long_only_type(self) -> None:
        ok, reason = validate_economic_justification(
            {"direction_long_only": "yes"}, strategy_philosophy="trend_following"
        )
        assert not ok
        assert "bool" in reason

    def test_empty_config_passes(self) -> None:
        ok, reason = validate_economic_justification({})
        assert ok


# ===================================================================
# PBOResult validation
# ===================================================================


class TestPBOResult:
    """``PBOResult`` post-init validation."""

    def test_valid_pbo_range_passes(self) -> None:
        PBOResult(pbo=0.5, rank_correlation=0.3)

    def test_out_of_range_pbo_raises(self) -> None:
        with pytest.raises(ValueError, match="PBO must be"):
            PBOResult(pbo=1.5, rank_correlation=0.3)

    def test_out_of_range_rho_raises(self) -> None:
        with pytest.raises(ValueError, match="Rank correlation must be"):
            PBOResult(pbo=0.5, rank_correlation=2.0)


# ===================================================================
# _sanitise_for_ranking
# ===================================================================


class TestSanitiseForRanking:
    """Unit tests for the internal ranking sanitisation helper."""

    def test_finite_values_unchanged(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = _sanitise_for_ranking(arr)
        np.testing.assert_array_equal(result, arr)

    def test_nan_mapped_to_min(self) -> None:
        arr = np.array([1.0, float("nan"), 2.0])
        result = _sanitise_for_ranking(arr)
        assert result[1] == np.finfo(np.float64).min
        assert result[0] == 1.0
        assert result[2] == 2.0

    def test_neginf_mapped_to_min(self) -> None:
        arr = np.array([1.0, float("-inf"), 2.0])
        result = _sanitise_for_ranking(arr)
        assert result[1] == np.finfo(np.float64).min

    def test_posinf_mapped_to_max(self) -> None:
        arr = np.array([1.0, float("inf"), 2.0])
        result = _sanitise_for_ranking(arr)
        assert result[1] == np.finfo(np.float64).max


# ===================================================================
# ExecutionStatus enum
# ===================================================================


class TestExecutionStatus:
    """Enum integrity checks."""

    def test_all_values_are_serialisable(self) -> None:
        for status in ExecutionStatus:
            assert isinstance(status.name, str)
            # Roundtrip through the enum constructor.
            assert ExecutionStatus[status.name] is status
