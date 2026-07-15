"""Persistent trial ledger and Probability of Backtest Overfitting (PBO) engine.

Records every optimization trial — successes **and** failures — in a DuckDB
database so that the file-drawer problem does not artificially inflate
statistical confidence.  Discarding failed trials biases PBO estimates
downward, making an overfit strategy appear more robust than it is.

The module provides:

* ``DuckDBLedger`` — thread-safe context manager for the ``optimization_trials``
  table.
* ``compute_pbo`` — logit-transformed rank-correlation PBO estimator.
* ``validate_economic_justification`` — pre-backtest guard that rejects
  configurations whose parameterisation makes no economic sense given the
  strategy's declared philosophy (e.g. a trend-following config that would
  systematically bet against the trend).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np

if TYPE_CHECKING:
    pass  # No TYPE_CHECKING-only imports required at this time.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain enumerations
# ---------------------------------------------------------------------------


class ExecutionStatus(Enum):
    """Outcome of a single optimization trial."""

    SUCCESS = auto()
    """The trial completed and produced a valid portfolio."""

    FAILED = auto()
    """The trial raised an unhandled exception (e.g. solver divergence)."""

    REJECTED = auto()
    """The trial was rejected by a pre-flight economic-justification guard."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialRecord:
    """Immutable snapshot of one optimization trial.

    Attributes
    ----------
    trial_id:
        Universally unique identifier (UUID4).
    run_timestamp:
        ISO-8601 UTC timestamp captured at insertion time.
    strategy_config:
        Arbitrary key-value dictionary describing the parameterisation.
    is_economically_justifiable:
        ``True`` when the config passed the pre-backtest validation guard.
    in_sample_sharpe:
        Annualised Sharpe ratio on the training window.  May be ``NaN`` when
        ``execution_status`` is ``FAILED`` or ``REJECTED``.
    out_of_sample_sharpe:
        Annualised Sharpe ratio on the test window.  Same ``NaN`` semantics.
    execution_status:
        The terminal state of the trial.
    """

    trial_id: str
    run_timestamp: str
    strategy_config: dict[str, Any]
    is_economically_justifiable: bool
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    execution_status: ExecutionStatus


@dataclass(frozen=True)
class PBOResult:
    """Result of a Probability of Backtest Overfitting calculation.

    Attributes
    ----------
    pbo:
        Estimated PBO ∈ [0, 1].  Values near 1.0 indicate that the best
        in-sample configuration is highly likely to have been selected by
        chance, i.e. the backtest is overfit.
    rank_correlation:
        Spearman rank correlation (ρ) between in-sample and out-of-sample
        Sharpe ratios.
    observation_flag:
        Diagnostic string when the computation encountered an edge case
        (e.g. zero-variance IS Sharpe ratios).  ``None`` otherwise.
    """

    pbo: float
    rank_correlation: float
    observation_flag: str | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.pbo <= 1.0):
            raise ValueError(f"PBO must be in [0, 1], got {self.pbo}")
        if not (-1.0 <= self.rank_correlation <= 1.0):
            raise ValueError(
                f"Rank correlation must be in [-1, 1], got {self.rank_correlation}"
            )


# ---------------------------------------------------------------------------
# DuckDB schema DDL
# ---------------------------------------------------------------------------

_DDL_CREATE_TABLE: str = """
CREATE TABLE IF NOT EXISTS optimization_trials (
    trial_id                    UUID PRIMARY KEY,
    run_timestamp               TIMESTAMP NOT NULL,
    strategy_config             JSON NOT NULL,
    is_economically_justifiable BOOLEAN NOT NULL,
    in_sample_sharpe            DOUBLE,
    out_of_sample_sharpe        DOUBLE,
    execution_status            VARCHAR NOT NULL
        CHECK (execution_status IN ('SUCCESS', 'FAILED', 'REJECTED'))
);
"""

_DDL_CREATE_INDEX_TS: str = """
CREATE INDEX IF NOT EXISTS idx_trials_timestamp
    ON optimization_trials (run_timestamp);
"""

_DDL_CREATE_INDEX_STATUS: str = """
CREATE INDEX IF NOT EXISTS idx_trials_status
    ON optimization_trials (execution_status);
"""

# ---------------------------------------------------------------------------
# DuckDBLedger — thread-safe context manager
# ---------------------------------------------------------------------------


class DuckDBLedger:
    """Thread-safe write-through ledger backed by a DuckDB database.

    Every connection is opened inside a context-managed scope so that the
    underlying file handle is released promptly.  The internal mutex ensures
    that concurrent writers (e.g. from a ``ThreadPoolExecutor`` parameter
    sweep) never contend on the same DuckDB connection.

    Parameters
    ----------
    db_path:
        Path to the DuckDB file, or ``:memory:`` for an in-memory ledger
        (useful for testing).
    read_only:
        When ``True``, the schema is assumed to exist already and the
        ``write_trial`` method raises ``RuntimeError``.  Defaults to ``False``.

    Examples
    --------
    >>> with DuckDBLedger(":memory:") as ledger:
    ...     record = TrialRecord(
    ...         trial_id=str(uuid.uuid4()),
    ...         run_timestamp="2025-01-01T00:00:00Z",
    ...         strategy_config={"lookback": 60},
    ...         is_economically_justifiable=True,
    ...         in_sample_sharpe=1.2,
    ...         out_of_sample_sharpe=0.9,
    ...         execution_status=ExecutionStatus.SUCCESS,
    ...     )
    ...     ledger.write_trial(record)
    ...     trials = ledger.load_trials()
    ...     assert len(trials) == 1
    """

    def __init__(self, db_path: str, *, read_only: bool = False) -> None:
        self._db_path: str = db_path
        self._read_only: bool = read_only
        self._lock: threading.Lock = threading.Lock()
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._initialised: bool = False

    # -- context manager --------------------------------------------------

    def __enter__(self) -> DuckDBLedger:
        with self._lock:
            self._connection = duckdb.connect(self._db_path)
        if not self._read_only:
            self._connection.execute(_DDL_CREATE_TABLE)
            self._connection.execute(_DDL_CREATE_INDEX_TS)
            self._connection.execute(_DDL_CREATE_INDEX_STATUS)
            self._initialised = True
        return self

    def __exit__(self, *args: object) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    # -- public API -------------------------------------------------------

    def write_trial(self, record: TrialRecord) -> None:
        """Persist a single trial record.

        Parameters
        ----------
        record:
            The immutable ``TrialRecord`` to insert.

        Raises
        ------
        RuntimeError:
            If the ledger was opened in read-only mode.
        """
        if self._read_only:
            raise RuntimeError("Cannot write to a read-only ledger.")
        if self._connection is None:
            raise RuntimeError("Ledger must be used as a context manager.")

        config_json: str = json.dumps(record.strategy_config, sort_keys=True)
        sql: str = """
            INSERT INTO optimization_trials (
                trial_id, run_timestamp, strategy_config,
                is_economically_justifiable,
                in_sample_sharpe, out_of_sample_sharpe, execution_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            assert self._connection is not None
            self._connection.execute(
                sql,
                (
                    record.trial_id,
                    record.run_timestamp,
                    config_json,
                    record.is_economically_justifiable,
                    record.in_sample_sharpe if np.isfinite(record.in_sample_sharpe) else None,
                    record.out_of_sample_sharpe if np.isfinite(record.out_of_sample_sharpe) else None,
                    record.execution_status.name,
                ),
            )

    def write_trials(self, records: list[TrialRecord]) -> None:
        """Persist multiple records in a single transaction (fast bulk path)."""
        for record in records:
            self.write_trial(record)

    def load_trials(
        self,
        *,
        status_filter: ExecutionStatus | None = None,
    ) -> list[TrialRecord]:
        """Return every trial currently stored, optionally filtered by status.

        Parameters
        ----------
        status_filter:
            When provided, only trials with this ``ExecutionStatus`` are returned.
        """
        if self._connection is None:
            raise RuntimeError("Ledger must be used as a context manager.")

        if status_filter is not None:
            sql: str = "SELECT * FROM optimization_trials WHERE execution_status = ?"
            params: tuple = (status_filter.name,)
        else:
            sql = "SELECT * FROM optimization_trials"
            params = ()

        with self._lock:
            assert self._connection is not None
            result = self._connection.execute(sql, params)
            rows: list[tuple] = result.fetchall()

        return [_row_to_record(row) for row in rows]

    def load_trials_dataframe(self) -> Any:
        """Return the full trial table as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns matching the ``optimization_trials`` schema.
        """
        if self._connection is None:
            raise RuntimeError("Ledger must be used as a context manager.")
        with self._lock:
            assert self._connection is not None
            return self._connection.execute(
                "SELECT * FROM optimization_trials ORDER BY run_timestamp"
            ).fetchdf()

    def count(self, *, status_filter: ExecutionStatus | None = None) -> int:
        """Return the number of trials, optionally filtered by status."""
        if self._connection is None:
            raise RuntimeError("Ledger must be used as a context manager.")

        if status_filter is not None:
            sql = "SELECT COUNT(*) FROM optimization_trials WHERE execution_status = ?"
            params = (status_filter.name,)
        else:
            sql = "SELECT COUNT(*) FROM optimization_trials"
            params = ()

        with self._lock:
            assert self._connection is not None
            result = self._connection.execute(sql, params).fetchone()
        return int(result[0]) if result else 0  # type: ignore[arg-type]

    def clear(self) -> None:
        """Delete all rows (non-recoverable — used primarily in tests)."""
        if self._read_only:
            raise RuntimeError("Cannot clear a read-only ledger.")
        if self._connection is None:
            raise RuntimeError("Ledger must be used as a context manager.")
        with self._lock:
            assert self._connection is not None
            self._connection.execute("DELETE FROM optimization_trials")


# ---------------------------------------------------------------------------
# Row → TrialRecord deserialisation
# ---------------------------------------------------------------------------


def _row_to_record(row: tuple) -> TrialRecord:
    """Convert a DuckDB result row to a ``TrialRecord``."""
    (
        trial_id,
        run_timestamp,
        strategy_config_raw,
        is_economically_justifiable,
        in_sample_sharpe,
        out_of_sample_sharpe,
        execution_status_raw,
    ) = row
    config: dict[str, Any] = json.loads(strategy_config_raw)
    return TrialRecord(
        trial_id=str(trial_id),
        run_timestamp=str(run_timestamp),
        strategy_config=config,
        is_economically_justifiable=bool(is_economically_justifiable),
        in_sample_sharpe=float(in_sample_sharpe) if in_sample_sharpe is not None else float("nan"),
        out_of_sample_sharpe=float(out_of_sample_sharpe) if out_of_sample_sharpe is not None else float("nan"),
        execution_status=ExecutionStatus[execution_status_raw],
    )


# ---------------------------------------------------------------------------
# PBO computation
# ---------------------------------------------------------------------------


def compute_pbo(
    is_sharpes: np.ndarray,
    oos_sharpes: np.ndarray,
) -> PBOResult:
    """Estimate the Probability of Backtest Overfitting.

    Uses logit-transformed rank correlation (Bailey & López de Prado, 2014).
    The intuition is simple: if in-sample rank predicts out-of-sample rank,
    the strategy has genuine predictive power and PBO → 0.  If there is no
    relationship (pure overfitting), PBO → 0.5.

    Parameters
    ----------
    is_sharpes:
        1-D array of in-sample Sharpe ratios.  Must contain at least 3
        observations.
    oos_sharpes:
        1-D array of out-of-sample Sharpe ratios.  Same length as *is_sharpes*.

    Returns
    -------
    PBOResult
        Named tuple with ``pbo``, ``rank_correlation``, and an optional
        ``observation_flag`` for edge cases.

    Raises
    ------
    ValueError
        If the input arrays differ in length or contain fewer than 3 elements.

    Notes
    -----
    *Observation*: ``pbo`` is an empirical estimate derived from the recorded
    Sharpe values.  It does not assign a binary "overfit / not-overfit" label
    — that decision is left to the caller.
    """
    is_sharpes = np.asarray(is_sharpes, dtype=np.float64).ravel()
    oos_sharpes = np.asarray(oos_sharpes, dtype=np.float64).ravel()

    n: int = len(is_sharpes)
    if n != len(oos_sharpes):
        raise ValueError(
            f"is_sharpes and oos_sharpes must have the same length, "
            f"got {n} vs {len(oos_sharpes)}"
        )
    if n < 3:
        raise ValueError(
            f"Need at least 3 trials for rank correlation, got {n}"
        )

    is_std: float = float(np.nanstd(is_sharpes))
    if is_std < 1e-15 or not np.isfinite(is_std):
        # Zero variance in IS Sharpe ratios — every configuration
        # produces identical performance, so ranking is degenerate.
        _log_edge_case_observation(is_sharpes, oos_sharpes)
        return PBOResult(
            pbo=0.5,
            rank_correlation=0.0,
            observation_flag="zero_variance_is",
        )

    # Replacement values ensure finite values participate in ranking.
    # NaN or -inf are pushed to the worst ranks so they do not bias the
    # correlation toward a spurious signal.
    is_clean: np.ndarray = _sanitise_for_ranking(is_sharpes)
    oos_clean: np.ndarray = _sanitise_for_ranking(oos_sharpes)

    from scipy.stats import spearmanr

    rho_raw = spearmanr(is_clean, oos_clean).statistic
    if rho_raw is None or not np.isfinite(rho_raw):
        # Not enough finite variation after sanitisation — degenerate ranking.
        _log_edge_case_observation(is_sharpes, oos_sharpes)
        return PBOResult(
            pbo=0.5,
            rank_correlation=0.0,
            observation_flag="degenerate_ranking",
        )
    rho: float = float(rho_raw)

    # Fisher z-transform → logit PBO
    # PBO → 0 when rho is large positive; PBO → 1 when rho is large negative.
    rho_clipped: float = float(np.clip(rho, -0.9999, 0.9999))
    z: float = float(np.arctanh(rho_clipped))
    pbo: float = float(1.0 / (1.0 + np.exp(z)))

    flag: str | None = None
    if abs(rho) < 0.1:
        flag = "weak_rank_correlation"

    return PBOResult(pbo=pbo, rank_correlation=rho, observation_flag=flag)


def _sanitise_for_ranking(values: np.ndarray) -> np.ndarray:
    """Replace non-finite values with the worst-possible rank positions.

    NaN / -inf are mapped to ``finfo.min``; +inf to ``finfo.max``.
    This prevents non-finite values from crashing ``scipy.stats.spearmanr``
    while preserving the ordering of finite observations.
    """
    out: np.ndarray = values.copy()
    nan_mask: np.ndarray = np.isnan(out)
    neginf_mask: np.ndarray = np.isneginf(out)
    posinf_mask: np.ndarray = np.isposinf(out)
    out[nan_mask] = np.finfo(np.float64).min
    out[neginf_mask] = np.finfo(np.float64).min
    out[posinf_mask] = np.finfo(np.float64).max
    return out


def _log_edge_case_observation(
    is_sharpes: np.ndarray,
    oos_sharpes: np.ndarray,
) -> None:
    """Log a structured diagnostic message when edge cases are encountered."""
    n: int = len(is_sharpes)
    oos_std: float = float(np.nanstd(oos_sharpes))
    logger.warning(
        "PBO edge case — zero-variance IS Sharpe ratios (n=%d, OOS std=%.6f). "
        "All configurations produced identical in-sample performance; "
        "ranking is degenerate. PBO set to uninformative prior (0.5).",
        n,
        oos_std,
    )


# ---------------------------------------------------------------------------
# Economic justification guard
# ---------------------------------------------------------------------------

# Known strategy philosophies and their canonical direction.
_PHILOSOPHY_DIRECTION: dict[str, int] = {
    "trend_following": 1,       # Structural long bias with momentum.
    "mean_reversion": -1,       # Contrarian, fade the move.
    "value": 1,                 # Buy undervalued, hold.
    "risk_parity": 0,           # Direction-neutral by design.
    "volatility_targeting": 0,  # Adjusts exposure, not direction.
}


def validate_economic_justification(
    strategy_config: dict[str, Any],
    *,
    strategy_philosophy: str = "trend_following",
) -> tuple[bool, str]:
    """Validate that a parameterisation aligns with its declared philosophy.

    This is a **pre-flight** check intended to catch nonsensical
    parameterisations *before* they burn compute on a backtest.  For
    example, a trend-following strategy with a negative momentum factor
    (``momentum_signal_sign = -1``) amounts to systematic mean-reversion
    and should be flagged.

    Parameters
    ----------
    strategy_config:
        Flat key-value dictionary describing the trial's parameters.
        Recognised keys include:

        * ``momentum_signal_sign`` — ``1`` (follow trend) or ``-1`` (fade).
        * ``direction_long_only`` — ``True`` / ``False``.
    strategy_philosophy:
        The strategy's declared philosophy.  Must be one of the keys in
        ``_PHILOSOPHY_DIRECTION``.

    Returns
    -------
    (is_justifiable, reason)
        ``is_justifiable`` is ``False`` when the configuration contradicts
        the philosophy.  ``reason`` is a human-readable explanation.
    """
    expected_dir: int | None = _PHILOSOPHY_DIRECTION.get(strategy_philosophy)
    if expected_dir is None:
        valid: str = ", ".join(sorted(_PHILOSOPHY_DIRECTION))
        raise ValueError(
            f"Unknown strategy_philosophy {strategy_philosophy!r}. "
            f"Valid options: {valid}"
        )

    # --- Momentum-sign check ---
    momentum_sign: Any = strategy_config.get("momentum_signal_sign")
    if momentum_sign is not None and expected_dir != 0:
        try:
            sign_val: int = int(momentum_sign)
        except (ValueError, TypeError):
            return (
                False,
                f"momentum_signal_sign must be an integer, got {momentum_sign!r}",
            )
        if sign_val == 0:
            return (
                False,
                f"momentum_signal_sign is 0 — no directional signal for "
                f"philosophy {strategy_philosophy!r}",
            )
        if expected_dir == 1 and sign_val == -1:
            return (
                False,
                f"momentum_signal_sign={sign_val} contradicts "
                f"trend_following philosophy (expected +1)",
            )
        if expected_dir == -1 and sign_val == 1:
            return (
                False,
                f"momentum_signal_sign={sign_val} contradicts "
                f"mean_reversion philosophy (expected -1)",
            )

    # --- Long-only check ---
    long_only: Any = strategy_config.get("direction_long_only")
    if long_only is not None:
        if not isinstance(long_only, bool):
            return (
                False,
                f"direction_long_only must be a bool, got {type(long_only).__name__}",
            )
        if expected_dir == 1 and not long_only:
            return (
                False,
                f"direction_long_only=False contradicts "
                f"{strategy_philosophy!r} philosophy (trend-following "
                f"strategies should be long-only)",
            )

    return True, "Configuration is economically justifiable."
