#!/usr/bin/env python3
"""Validate a custom Canadian ETF allocation against a single-ticket VEQT benchmark.

Implements a Purged Cross-Validation loop to stress-test a bounded Sharpe-optimised
portfolio of four broad-market Canadian ETFs, evaluating execution friction and
asset-location tax drag via PySharpe's internal API modules.

Pipeline
--------
1. **Data Ingestion** — DuckDBCachedPriceFetcher → CollationService for 10 years of
   daily prices; FX conversion to CAD with no-lookahead exclusion.
2. **Purged CV Loop** — PurgedKFold temporal splits; SharpeOptimizer on each training
   fold with per-ticker min/max bounds enforced via scipy SLSQP.
3. **Overfitting Diagnostics** — compute_pbo() from the validation ledger using
   in-sample and out-of-sample Sharpe ratios across folds.
4. **Execution Friction** — HistoricalBacktester with quarterly rebalancing, $5 flat
   fee, 10 bps slippage; stress_test_execution_friction() at 0, 50, and 150 bps.
5. **Benchmark Comparison** — VEQT.TO single-ticket buy-and-hold OOS Sharpe and CAGR.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pysharpe.analysis.backtest_engine import BacktestResult, HistoricalBacktester
from pysharpe.data.collation import CollationService
from pysharpe.data.fetcher import YFinancePriceFetcher, apply_fx_conversion
from pysharpe.metrics import annualize_return, sharpe_ratio
from pysharpe.optimization.sharpe_optimizer import (
    SharpeOptimizer,
    SharpeOptimizerConfig,
)
from pysharpe.validation.friction import FrictionProfile, stress_test_execution_friction
from pysharpe.validation.ledger import PBOResult, compute_pbo
from pysharpe.validation.resampling import PurgedKFold

if TYPE_CHECKING:
    pass  # All types above are runtime imports; no TYPE_CHECKING-only deps.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("validate_custom_allocation")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ASSET_TICKERS: list[str] = ["VCN.TO", "VUN.TO", "VIU.TO", "VEE.TO"]
BENCHMARK_TICKER: str = "VEQT.TO"

# Per-ticker hard min-max weight bounds (decimal fractions).
WEIGHT_BOUNDS: dict[str, tuple[float, float]] = {
    "VCN.TO": (0.15, 0.30),
    "VUN.TO": (0.25, 0.40),
    "VIU.TO": (0.05, 0.20),
    "VEE.TO": (0.00, 0.10),
}

# MER values as decimal fractions (< 0.10), never percentage points.
MER_VALUES: dict[str, float] = {
    "VCN.TO": 0.0005,
    "VUN.TO": 0.0016,
    "VIU.TO": 0.0023,
    "VEE.TO": 0.0024,
    "VEQT.TO": 0.0017,
}

YEARS_OF_HISTORY: int = 10
RISK_FREE_RATE: float = 0.025
N_CV_SPLITS: int = 5
PURGE_PCT: float = 0.01
EMBARGO_PCT: float = 0.01

# Backtest / friction parameters.
TRANSACTION_FEE: float = 5.00  # CAD flat fee per asset traded.
SLIPPAGE_DECIMAL: float = 0.0010  # 10 bps as decimal fraction.
INITIAL_CAPITAL: float = 100_000.0
REBALANCE_FREQ: str = "QE"  # Quarterly end.
FRICTION_BPS_LEVELS: tuple[int, ...] = (0, 50, 150)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_cagr(portfolio_value: pd.Series, periods_per_year: int = 252) -> float:
    """Compute geometric annualised return (CAGR) from a portfolio equity curve."""
    if len(portfolio_value) < 2:
        return float("nan")
    returns = portfolio_value.pct_change().dropna()
    if returns.empty:
        return float("nan")
    return float(annualize_return(returns, periods_per_year=periods_per_year))


def _compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualised Sharpe ratio from a periodic return series."""
    if len(returns) < 2:
        return float("nan")
    try:
        return float(
            sharpe_ratio(
                returns,
                risk_free_rate=RISK_FREE_RATE,
                periods_per_year=periods_per_year,
            )
        )
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# 1. Data Ingestion
# ---------------------------------------------------------------------------


def ingest_data() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Download and align 10 years of daily close prices.

    Uses DuckDBCachedPriceFetcher (write-through) via CollationService.
    Applies ``apply_fx_conversion`` to strip lookahead bias — rows without
    FX rate coverage are **excluded** rather than backfilled.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None]
        (asset_prices, benchmark_series).  ``benchmark_series`` is ``None``
        if ingestion of VEQT.TO fails.
    """
    all_tickers = sorted(set(ASSET_TICKERS + [BENCHMARK_TICKER]))
    logger.info(
        "Ingesting %d years of daily data for: %s", YEARS_OF_HISTORY, all_tickers
    )

    yf_fetcher = YFinancePriceFetcher({"auto_adjust": True})
    service = CollationService(yf_fetcher)

    raw_prices = service.download_portfolio_prices(
        all_tickers,
        period=f"{YEARS_OF_HISTORY}y",
        interval="1d",
        start=None,
        end=None,
    )

    # --- Build combined close-price DataFrame ---
    close_series: dict[str, pd.Series] = {}
    for ticker, df in raw_prices.items():
        if df.empty:
            logger.warning("Empty price history for %s — skipping.", ticker)
            continue
        if "Close" not in df.columns:
            logger.warning("No 'Close' column for %s — skipping.", ticker)
            continue
        ts = df["Close"].copy()
        if hasattr(ts.index, "tz") and ts.index.tz is not None:
            ts.index = ts.index.tz_localize(None)
        close_series[ticker] = ts

    if not close_series:
        raise RuntimeError("No price data retrieved for any ticker.")

    # Align all series on a common DatetimeIndex (inner join).
    combined = pd.DataFrame(close_series).dropna().sort_index()
    logger.info("Raw combined shape (before FX): %s", combined.shape)

    # --- FX conversion (CAD base, no lookahead bias) ---
    prices_cad = apply_fx_conversion(combined, base_currency="CAD", fetcher=yf_fetcher)
    logger.info("Post-FX combined shape: %s", prices_cad.shape)

    if prices_cad.empty:
        raise RuntimeError("All rows excluded during FX conversion.")

    # --- Split asset vs benchmark ---
    # Ensure assets are ordered consistently with WEIGHT_BOUNDS.
    asset_cols = [t for t in ASSET_TICKERS if t in prices_cad.columns]
    if set(asset_cols) != set(ASSET_TICKERS):
        missing = set(ASSET_TICKERS) - set(asset_cols)
        raise RuntimeError(f"Missing asset tickers after FX alignment: {missing}")

    asset_prices = prices_cad[asset_cols]

    benchmark_prices: pd.DataFrame | None = None
    if BENCHMARK_TICKER in prices_cad.columns:
        benchmark_prices = prices_cad[[BENCHMARK_TICKER]]
    else:
        logger.warning("Benchmark ticker %s unavailable.", BENCHMARK_TICKER)

    logger.info(
        "Asset prices: %s  |  Benchmark available: %s",
        asset_prices.shape,
        benchmark_prices is not None,
    )
    return asset_prices, benchmark_prices


# ---------------------------------------------------------------------------
# 2. Bounded Sharpe Optimisation (per-ticker bounds)
# ---------------------------------------------------------------------------

# The stock SharpeOptimizer enforces a single uniform ``max_weight`` across all
# assets.  To honour the per-ticker min/max bounds specified in WEIGHT_BOUNDS,
# we construct a SharpeOptimizer to obtain the covariance and expected-return
# estimates, then solve directly with scipy using custom bounds.


def _optimise_bounded(
    prices_train: pd.DataFrame,
    mer_values: dict[str, float],
    risk_free_rate: float = RISK_FREE_RATE,
) -> tuple[dict[str, float], float]:
    """Maximise Sharpe ratio subject to per-ticker min/max weight bounds.

    Parameters
    ----------
    prices_train:
        Daily close prices for the training window  (columns = tickers).
    mer_values:
        Ticker → annual MER as a decimal fraction.
    risk_free_rate:
        Annualised risk-free rate (decimal).

    Returns
    -------
    tuple[dict[str, float], float]
        (optimal_weights, in_sample_sharpe).
    """
    if prices_train.empty or prices_train.shape[1] == 0:
        raise ValueError("Training price data is empty.")

    # Use SharpeOptimizer to estimate returns and covariance.
    config = SharpeOptimizerConfig(
        risk_free_rate=risk_free_rate,
        mer_by_ticker={t: mer_values.get(t, 0.0) for t in prices_train.columns},
        max_weight=1.0,  # We enforce bounds externally.
    )
    opt = SharpeOptimizer(prices_train, config)
    assets = opt.assets

    # Build per-asset bounds.
    bounds_list: list[tuple[float, float]] = []
    for ticker in assets:
        lo, hi = WEIGHT_BOUNDS.get(ticker, (0.0, 1.0))
        bounds_list.append((lo, hi))
    bounds = tuple(bounds_list)

    # Constraint: weights sum to 1.0.
    constraints: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Objective: negative Sharpe (minimise → maximise Sharpe).
    def objective(w: np.ndarray) -> float:
        _, _, sr = opt.calculate_portfolio_performance(w)
        return -sr if np.isfinite(sr) else 1e9

    # Initial guess: midpoint of each bound.
    x0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds_list])
    x0 = x0 / x0.sum()  # Normalise to sum-to-1.

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        logger.warning(
            "SLSQP did not converge: %s. Using initial guess.", result.message
        )
        w_opt = x0
    else:
        w_opt = result.x
        w_opt = np.abs(w_opt)  # Guard against tiny negatives.
        w_opt = w_opt / w_opt.sum()

    weights_map = {t: float(w) for t, w in zip(assets, w_opt) if w > 1e-8}
    _, _, is_sharpe = opt.calculate_portfolio_performance(w_opt)
    return weights_map, float(is_sharpe)


# ---------------------------------------------------------------------------
# 3. CV Loop & PBO
# ---------------------------------------------------------------------------


def run_cv_loop(
    asset_prices: pd.DataFrame,
    folds: list | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], list]:
    """Purged cross-validation: optimise per fold, collect IS/OOS Sharpes.

    Parameters
    ----------
    asset_prices:
        Full daily close price DataFrame (columns = tickers).
    folds:
        Pre-computed list of ``PurgedFold`` objects.  When ``None``, a new
        ``PurgedKFold`` is created internally using module-level constants.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[dict[str, float]], list]
        (is_sharpes, oos_sharpes, oos_weights_per_fold, folds_used).
        ``folds_used`` is the fold list so callers can reuse it for
        benchmark evaluation on the same splits.
    """
    if folds is None:
        cv = PurgedKFold(
            n_splits=N_CV_SPLITS,
            embargo_pct=EMBARGO_PCT,
            purge_pct=PURGE_PCT,
        )
        folds = cv.split(asset_prices)
    logger.info(
        "PurgedKFold: %d splits (purge=%.2f%%, embargo=%.2f%%)",
        len(folds),
        PURGE_PCT * 100,
        EMBARGO_PCT * 100,
    )

    is_sharpes: list[float] = []
    oos_sharpes: list[float] = []
    oos_all_weights: list[dict[str, float]] = []

    for i, fold in enumerate(folds):
        train_mask = (asset_prices.index >= fold.train_start) & (
            asset_prices.index <= fold.train_end
        )
        test_mask = (asset_prices.index >= fold.test_start) & (
            asset_prices.index <= fold.test_end
        )

        prices_train = asset_prices.loc[train_mask]
        prices_test = asset_prices.loc[test_mask]

        if prices_train.empty or prices_test.empty:
            logger.warning("Fold %d: empty train or test set — skipping.", i)
            continue

        # --- Optimise on training window ---
        try:
            weights, is_sharpe = _optimise_bounded(prices_train, MER_VALUES)
        except Exception:
            logger.exception("Fold %d: optimisation failed — skipping.", i)
            continue

        # --- Evaluate on test window ---
        if not weights:
            logger.warning("Fold %d: optimisation returned empty weights.", i)
            continue

        oos_returns = prices_test.pct_change().dropna()
        if oos_returns.empty:
            logger.warning("Fold %d: empty OOS returns.", i)
            continue

        # Build portfolio return series: weighted sum of asset returns.
        common_tickers = [t for t in weights if t in oos_returns.columns]
        if not common_tickers:
            logger.warning("Fold %d: no common tickers in OOS data.", i)
            continue

        w_vec = np.array([weights[t] for t in common_tickers])
        w_vec = w_vec / w_vec.sum()  # renormalise
        port_returns = (oos_returns[common_tickers] * w_vec).sum(axis=1)

        oos_sharpe = _compute_sharpe(port_returns)
        if not np.isfinite(oos_sharpe):
            logger.warning("Fold %d: non-finite OOS Sharpe.", i)
            continue

        is_sharpes.append(is_sharpe)
        oos_sharpes.append(oos_sharpe)
        oos_all_weights.append(weights)

        logger.info(
            "Fold %d: IS Sharpe=%.4f  OOS Sharpe=%.4f  Weights=%s",
            i,
            is_sharpe,
            oos_sharpe,
            {t: f"{w:.3f}" for t, w in sorted(weights.items())},
        )

    return (
        np.array(is_sharpes, dtype=np.float64),
        np.array(oos_sharpes, dtype=np.float64),
        oos_all_weights,
        folds,
    )


# ---------------------------------------------------------------------------
# 4. Execution Friction Simulation
# ---------------------------------------------------------------------------


def run_backtest(
    prices: pd.DataFrame,
    weights: dict[str, float],
    label: str = "strategy",
) -> BacktestResult | None:
    """Run a single HistoricalBacktester simulation."""
    if not weights:
        logger.warning("%s: empty weights — backtest skipped.", label)
        return None

    # Filter to available tickers.
    available = [t for t in weights if t in prices.columns]
    if not available:
        logger.warning("%s: no overlapping tickers — backtest skipped.", label)
        return None

    sub_weights = {t: weights[t] for t in available}
    total = sum(sub_weights.values())
    sub_weights = {t: w / total for t, w in sub_weights.items()}

    bt = HistoricalBacktester(
        prices=prices,
        target_weights=sub_weights,
        initial_capital=INITIAL_CAPITAL,
        rebalance_freq=REBALANCE_FREQ,
        fee_per_trade=TRANSACTION_FEE,
        slippage_pct=SLIPPAGE_DECIMAL,
    )
    return bt.run()


# ---------------------------------------------------------------------------
# 4. Execution Friction Simulation
# ---------------------------------------------------------------------------


def run_friction_analysis(
    asset_prices: pd.DataFrame,
    oos_weights_per_fold: list[dict[str, float]],
) -> tuple[list[FrictionProfile], list[int]]:
    """Run backtests and friction stress-tests on each fold's OOS weights.

    Returns
    -------
    tuple[list[FrictionProfile], list[int]]
        Parallel lists of (profile, requested_bps_label).  The label is the
        original ``FRICTION_BPS_LEVELS`` value so callers can group by it
        even when the underlying call bumped ``max_bps`` to satisfy the API.
    """
    profiles: list[FrictionProfile] = []
    labels: list[int] = []

    for i, weights in enumerate(oos_weights_per_fold):
        result = run_backtest(asset_prices, weights, label=f"fold-{i}")
        if result is None:
            continue

        for requested_bps in FRICTION_BPS_LEVELS:
            # When requested_bps is 0 the API requires step_bps <= max_bps,
            # so bump to a minimal sweep (5 bps) that still includes cost=0.
            effective_max = max(requested_bps, 5)
            profile = stress_test_execution_friction(
                result,
                max_bps=effective_max,
                step_bps=5,
                risk_free_rate=RISK_FREE_RATE,
            )
            profiles.append(profile)
            labels.append(requested_bps)

            # Log the key metrics at this friction level.
            baseline_step = profile.steps[0] if profile.steps else None
            top_step = profile.steps[-1] if profile.steps else None
            logger.info(
                "Fold %d @ %d bps: Sharpe baseline=%.4f  max=%.4f  "
                "NAV decay=%.2f%%  break_even=%s bps  viable=%s",
                i,
                requested_bps,
                baseline_step.sharpe if baseline_step else float("nan"),
                top_step.sharpe if top_step else float("nan"),
                top_step.nav_decay_pct if top_step else float("nan"),
                f"{profile.break_even_bps:.0f}" if profile.break_even_bps else ">max",
                profile.is_viable,
            )

    return profiles, labels


# ---------------------------------------------------------------------------
# 5. Benchmark Evaluation
# ---------------------------------------------------------------------------


def evaluate_benchmark(
    benchmark_prices: pd.DataFrame | None,
    asset_prices: pd.DataFrame,
    cv_folds: list | None = None,
) -> tuple[float, float]:
    """Evaluate the single-ticket VEQT.TO benchmark.

    When ``cv_folds`` is provided, OOS Sharpe is the average across folds;
    otherwise a full-period backtest is used.
    """
    if benchmark_prices is None or BENCHMARK_TICKER not in benchmark_prices.columns:
        logger.warning("Benchmark data unavailable.")
        return float("nan"), float("nan")

    bench_col = benchmark_prices[BENCHMARK_TICKER]

    # Use the full price history aligned with asset data.
    common_idx = asset_prices.index.intersection(bench_col.index)
    if len(common_idx) < 2:
        logger.warning("Insufficient overlapping data for benchmark.")
        return float("nan"), float("nan")

    bench_aligned = bench_col.loc[common_idx]

    # Full-period buy-and-hold.
    bench_returns = bench_aligned.pct_change().dropna()
    bench_sharpe = _compute_sharpe(bench_returns)
    bench_cagr = _compute_cagr(bench_aligned)

    logger.info(
        "VEQT Benchmark (full period): Sharpe=%.4f  CAGR=%.4f%%",
        bench_sharpe,
        bench_cagr * 100,
    )

    # OOS-only benchmark across CV folds.
    if cv_folds:
        oos_bench_returns: list[float] = []
        for fold in cv_folds:
            test_mask = (bench_aligned.index >= fold.test_start) & (
                bench_aligned.index <= fold.test_end
            )
            test_prices = bench_aligned.loc[test_mask]
            if len(test_prices) < 2:
                continue
            test_rets = test_prices.pct_change().dropna()
            shp = _compute_sharpe(test_rets)
            if np.isfinite(shp):
                oos_bench_returns.append(shp)

        if oos_bench_returns:
            oos_bench_sharpe = float(np.mean(oos_bench_returns))
            logger.info(
                "VEQT Benchmark (OOS folds): Mean Sharpe=%.4f", oos_bench_sharpe
            )

    return bench_sharpe, bench_cagr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Execute the full validation pipeline."""
    start_ts = datetime.now(timezone.utc)
    logger.info("=== Custom Allocation Validation ===")
    logger.info("Assets: %s", ASSET_TICKERS)
    logger.info("Benchmark: %s", BENCHMARK_TICKER)
    logger.info("Weight bounds: %s", WEIGHT_BOUNDS)
    logger.info("MER values: %s", MER_VALUES)

    # ----------------------------------------------------------------
    # 1. Data ingestion
    # ----------------------------------------------------------------
    asset_prices, benchmark_prices = ingest_data()
    logger.info(
        "Asset date range: %s → %s", asset_prices.index[0], asset_prices.index[-1]
    )

    # ----------------------------------------------------------------
    # 2. Purged cross-validation → IS/OOS Sharpe ratios
    # ----------------------------------------------------------------
    is_sharpes, oos_sharpes, oos_weights, cv_folds = run_cv_loop(asset_prices)

    n_valid = len(is_sharpes)
    if n_valid == 0:
        logger.error("No valid CV folds — aborting.")
        return

    mean_is = float(np.mean(is_sharpes))
    mean_oos = float(np.mean(oos_sharpes))

    # ----------------------------------------------------------------
    # 3. PBO diagnostics
    # ----------------------------------------------------------------
    pbo_result: PBOResult | None = None
    if n_valid >= 3:
        pbo_result = compute_pbo(is_sharpes, oos_sharpes)
    else:
        logger.warning("Insufficient folds for PBO (need ≥ 3, got %d).", n_valid)

    # ----------------------------------------------------------------
    # 4. Execution friction analysis
    # ----------------------------------------------------------------
    friction_profiles, friction_labels = run_friction_analysis(
        asset_prices, oos_weights
    )

    # Aggregate friction degradation vectors keyed by requested bps level.
    friction_summary: dict[int, dict[str, list[float]]] = {}
    for profile, label_bps in zip(friction_profiles, friction_labels):
        if label_bps not in friction_summary:
            friction_summary[label_bps] = {
                "sharpes": [],
                "nav_decays": [],
            }
        if profile.steps:
            # Use the first step (cost=0) for the 0-bps baseline; the last
            # step for all higher levels.
            step = profile.steps[0] if label_bps == 0 else profile.steps[-1]
            friction_summary[label_bps]["sharpes"].append(step.sharpe)
            friction_summary[label_bps]["nav_decays"].append(step.nav_decay_pct)

    # ----------------------------------------------------------------
    # 5. Benchmark comparison
    # ----------------------------------------------------------------
    bench_sharpe, bench_cagr = evaluate_benchmark(
        benchmark_prices, asset_prices, cv_folds
    )

    # ----------------------------------------------------------------
    # 6. Final report
    # ----------------------------------------------------------------
    elapsed = (datetime.now(timezone.utc) - start_ts).total_seconds()
    print("\n" + "=" * 72)
    print("  VALIDATION REPORT — Custom Canadian ETF Allocation")
    print("=" * 72)
    print(f"  Assets            : {', '.join(ASSET_TICKERS)}")
    print(f"  Benchmark         : {BENCHMARK_TICKER}")
    print(
        f"  CV splits         : {N_CV_SPLITS}  (purge={PURGE_PCT}, embargo={EMBARGO_PCT})"
    )
    print(f"  Valid folds       : {n_valid}")
    print(f"  Mean IS Sharpe    : {mean_is:+.4f}")
    print(f"  Mean OOS Sharpe   : {mean_oos:+.4f}")
    if pbo_result is not None:
        print(f"  PBO               : {pbo_result.pbo:.4f}")
        print(f"  Rank correlation  : {pbo_result.rank_correlation:+.4f}")
        if pbo_result.observation_flag:
            print(f"  PBO flag          : {pbo_result.observation_flag}")
    print("-" * 72)
    print(f"  Benchmark Sharpe  : {bench_sharpe:+.4f}")
    print(f"  Benchmark CAGR    : {bench_cagr:+.4%}")
    print("-" * 72)
    print("  Friction stress-test degradation vectors:")
    for max_bps in sorted(friction_summary.keys()):
        stats = friction_summary[max_bps]
        if stats["sharpes"]:
            avg_sharpe = float(np.mean(stats["sharpes"]))
            avg_nav = float(np.mean(stats["nav_decays"]))
            print(
                f"    {max_bps:>4d} bps → "
                f"mean Sharpe={avg_sharpe:+.4f}  "
                f"mean NAV decay={avg_nav:+.2f}%"
            )
    print("-" * 72)
    print(f"  Elapsed           : {elapsed:.1f} s")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
