# Gotchas

Known failure patterns discovered during development. Every entry represents a
bug that shipped and was later fixed — the goal is to prevent recurrence.

---

### 2026 — Equal-weight collapse from near-singular sample covariance
- **Symptom**: VFV.TO, VDY.TO, and VIU.TO all received identical optimized
  weights (e.g. all 0.20 or all 0.25) despite different risk profiles.
- **Root cause**: Sample covariance matrices for highly correlated broad-market
  ETFs become near-singular (rank-deficient).  Two cascading failures result:
  1. ``shrinkage_expected_return()`` used sample covariance for the Bayes-Stein
     computation.  The near-singular matrix caused a ``LinAlgError``, which fell
     back to the grand mean for every asset — producing identical expected
     returns and robbing the optimizer of any differentiation signal.
  2. The global ``max_weight`` ceiling (default 0.20) further constrained the
     feasible region.  With ``max_weight * n_assets ≈ 1.0``, the simplex
     collapsed to a single equal-weight point, leaving zero degrees of freedom
     even after the covariance issue was resolved.
- **Fix**:
  1. Replaced sample covariance with **Ledoit-Wolf shrunk covariance** for the
     shrinkage-intensity matrix inside ``shrinkage_expected_return()``.  The
     shrunk estimate is guaranteed positive-definite, preventing the
     ``LinAlgError`` and eliminating the grand-mean fallback.
  2. Removed the global ``max_weight`` artificial ceiling entirely (default now
     ``1.0`` / 100%).  The optimizer operates with full degrees of freedom inside
     the weight simplex.  Concentration risk is handled natively by the
     Ledoit-Wolf shrinkage already applied to the covariance.
  3. Removed the ``1/(n-2) + 0.01`` slack heuristic — it was mathematically
     unsafe for small portfolios (N=2 → ``ZeroDivisionError``) and
     philosophically wrong: an optimizer should not silently override the
     user's stated constraints.
- **Regression test**: ``test_optimise_from_prices_converges_on_differentiated_weights``
- **Grep guard**: ``grep -rn 'shrinkage_expected_return\|LedoitWolf' src/pysharpe/portfolio_optimization.py``

### 2026 — int object has no attribute 'date' in HistoryLinker
- **Symptom**: ``AttributeError: 'int' object has no attribute 'date'`` in
  ``linkage.py:357`` when stitching proxy data. Crashed the
  `process_all_portfolios` workflow.
- **Root cause**: ``common_dates.min()`` returns an ``int`` when the index is
  integer-based (e.g. in tests with stub fetchers).  ``t0.date()`` was called
  unconditionally assuming ``t0`` would always be a ``pd.Timestamp``.
- **Fix**: Guard ``t0.date()`` with ``hasattr(t0, "date")``, falling back to
  ``str(t0)`` for display.
- **Regression test**: ``test_process_all_portfolios_uses_repository``
- **Grep guard**: ``grep -rn '\.date()' src/pysharpe/data/linkage.py``

### 2026 — Silent equal-weight fallbacks in optimizer execution paths
- **Symptom**: When `scipy.optimize.minimize` or `EfficientFrontier` failed
  (singular covariance, infeasible constraints, solver non-convergence), the
  optimizer silently returned equal-weight allocations.  Users saw plausible-looking
  weights with normal Sharpe ratios, masking the underlying failure.
- **Root cause**: Both `SharpeOptimizer.optimize()` and `BayesianOptimizer.optimize()`
  had `if not result.success: return _fallback_result()` patterns.  The
  `validate_custom_allocation.py` script had its own `if not result.success: w_opt = x0`
  branch.  `optimise_from_prices` silently fell back to sample covariance when
  scikit-learn was missing.
- **Fix**:
  1. All `scipy` failure paths now raise ``RuntimeError`` with ``result.message``.
  2. `SharpeOptimizer._estimate_covariance()` now uses ``sklearn.covariance.LedoitWolf``
     to guarantee PSD covariance estimates.
  3. `optimise_from_prices` raises ``RuntimeError`` when Ledoit-Wolf fails
     (no sample-covariance fallback).
  4. ``_fallback_result()`` was removed entirely.
- **Regression test**: ``test_sharpe_optimizer_optimize_raises_on_failure``
- **Grep guard**: ``grep -rn '_fallback_result\|if not result.success' src/pysharpe/ --include='*.py'``

### 2024 — MER double-division
- **Symptom**: Reported portfolio MER was 100× smaller than intended. An ETF
  with a 0.17% MER showed as 0.0017%.
- **Root cause**: MER values already stored as decimal fractions (0.0017) were
  divided by 100 a second time inside `SharpeOptimizer` and
  `optimise_portfolio_for_sharpe`.
- **Fix**: Removed the redundant `/ 100` division. Default VEQT MER corrected
  from `0.17` (percentage point) to `0.0017` (decimal fraction).
- **Regression test**: `test_sharpe_optimizer_mer_deduction_is_decimal_not_percentage`
- **Grep guard**: `grep -rn '/ 100' src/pysharpe/ --include='*.py' | grep -i mer`

### 2024 — FX lookahead bias from bfill
- **Symptom**: Future exchange rates were applied to historical prices,
  inflating backtest returns with information that wasn't available at the time.
- **Root cause**: `apply_fx_conversion` called `.bfill()` after `.ffill()` on
  aligned exchange-rate series.
- **Fix**: Removed `.bfill()`. Leading rows without FX coverage are now
  detected and excluded with a warning. If all rows are excluded, a
  `ValueError` is raised.
- **Regression test**: `test_apply_fx_conversion_excludes_rows_with_no_fx_data`
- **Grep guard**: `grep -rn 'bfill()' src/pysharpe/ --include='*.py'`

### 2024 — Stale LRU cache on collated CSVs
- **Symptom**: After re-downloading prices, the optimizer silently used the old
  collated data because the LRU cache key didn't include file modification time.
- **Root cause**: `_cached_collated_prices` was keyed on `(portfolio_name,
  collated_dir, time_constraint)` only.
- **Fix**: Added `csv_path.stat().st_mtime` as a fourth cache key, forcing a
  cache miss whenever the file is overwritten.
- **Regression test**: `test_load_collated_prices_reflects_updated_file`
- **Grep guard**: `grep -rn '@lru_cache' src/pysharpe/ --include='*.py'`

### 2024 — DuckDB wrapping custom fetchers
- **Symptom**: Test stubs were silently bypassed because the DuckDB cache
  already held data for the requested ticker from a previous session.
- **Root cause**: `CollationService` wrapped every non-`DuckDBCachedPriceFetcher`
  fetcher in the write-through cache, including test stubs and custom
  implementations.
- **Fix**: DuckDB wrapping is now applied only to `YFinancePriceFetcher`
  instances. Custom fetchers pass through directly.
- **Regression test**: `test_collation_service_uses_settings_cache_dir` (verifies
  wrapping is conditional)
- **Grep guard**: `grep -rn 'DuckDBCachedPriceFetcher' src/pysharpe/ --include='*.py'`

### 2024 — Infeasible geo constraint on missing regions
- **Symptom**: `"infeasible solver"` crash when a portfolio had no assets mapped
  to a region with a lower-bound constraint.
- **Root cause**: Geographic lower-bound constraints were applied blindly to all
  configured regions, even those absent from the portfolio.
- **Fix**: The optimizer now drops lower-bound constraints for regions that
  contain no mapped assets.
- **Regression test**: `test_optimise_portfolio_respects_constraints` (in
  `test_portfolio_optimization.py`)
- **Grep guard**: `grep -rn 'lower_bound' src/pysharpe/ --include='*.py'`

### 2025 — Hard-coded three-pillar blend overwriting 60/40 baseline
- **Symptom**: When tax characteristics were omitted or all assets targeted the
  same account, the composite score still included a 0.2-weight tax-efficiency
  pillar, diluting the core 60/40 investment-heuristic signal.
- **Root cause**: `score_opportunities` always blended using the three config
  weights (`weight_underweight`, `weight_valuation`, `weight_tax_efficiency`)
  without checking whether tax location was differentiable.
- **Fix**: Added `_is_tax_location_differentiable()` guard. When tax
  characteristics are empty or all rows share a single `target_account`,
  the blend collapses to the authoritative 60/40 (`_CORE_UNDERWEIGHT_WEIGHT` /
  `_CORE_VALUATION_WEIGHT`) bypassing tax entirely.
- **Regression test**: `test_tax_neutral_scales_to_60_40`,
  `test_uniform_account_scales_to_60_40`, `test_mixed_accounts_uses_three_pillar_blend`
- **Grep guard**: `grep -rn '_CORE_UNDERWEIGHT_WEIGHT\|_CORE_VALUATION_WEIGHT\|_is_tax_location_differentiable' src/pysharpe/execution/allocator.py`

### 2024 — pandas 2.2+ groupby(axis=1) deprecation
- **Symptom**: Warning/crash in the collation layer on pandas ≥ 2.2.
- **Root cause**: `groupby(axis=1)` was deprecated in newer pandas versions.
- **Fix**: Replaced with transposed operations or column-wise iteration.
- **Regression test**: Covered by existing collation tests.
- **Grep guard**: `grep -rn 'groupby.*axis=1' src/pysharpe/ --include='*.py'`

### 2024 (superseded) — max_weight too restrictive for small portfolios
- **Symptom**: `ValueError: The max_weight constraint (0.2) is too restrictive
  for 4 assets to sum to 1.0.` Small portfolios (≤ 4 assets) with the default
  `max_weight=0.20` were infeasible because 4 × 0.20 = 0.80 < 1.0.
- **Root cause**: Both `optimise_from_prices` and
  `optimise_portfolio_for_sharpe` raised a hard `ValueError` when
  `max_weight * n_assets < 1.0` instead of auto-adjusting.
- **Historical fix (2024)**: Replaced `ValueError` with a `logger.warning` and
  auto-adjusted `max_weight` to `1.0 / n_assets`.
- **Final fix (2026)**: The auto-adjust was an artificial hack that masked the
  deeper problem (see **2026 — Equal-weight collapse** above).  The default
  `max_weight` is now ``1.0`` (unconstrained) across the entire pipeline, and
  the auto-adjust logic — including the unsafe ``1/(n-2) + 0.01`` variant — has
  been removed entirely.  Concentration risk is handled natively by
  Ledoit-Wolf covariance shrinkage.

---

## Pre-commit verification checklist

Before declaring any change "done", verify:

- [ ] All MER values are decimal fractions (< 0.10)?
- [ ] No `.bfill()` on time-series data?
- [ ] FX conversion excludes rows without rate coverage instead of backfilling?
- [ ] `AssetTaxCharacteristics` income fractions sum to 1.0?
- [ ] Caches invalidated properly (mtime keys, `cache_clear()` in tests)?
- [ ] New public symbols registered in `_EXPORT_MAP`, `TYPE_CHECKING`, AND `__all__`?
- [ ] Tests use synthetic data only, with fixed seeds?
- [ ] No new dependencies on network calls, day-trading, or price prediction?
