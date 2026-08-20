# Gotchas

### 2026 — Walk-forward NaN handling: fully-missing columns vs row gaps

- Symptom: a permanently-missing ticker in the price frame wiped out the
  whole backtest (row-wise ``dropna()`` removed every observation), while
  occasional NaN gaps silently shifted window boundaries.
- Fix: ``WalkForwardBacktester.run`` drops columns that are all-NaN with a
  warning (they cannot be traded), then applies listwise row deletion.  All
  columns missing → ``ValueError("No usable price data…")``.
- Regression tests: ``tests/test_analysis_backtest_engine.py::TestWalkForwardMissingData``.

### 2026 — pytest-cov dotted module form breaks the duckdb shim

- Symptom: ``pytest --cov=pysharpe.validation.metrics`` (module-dotted form)
  fails at collection with ``ModuleNotFoundError: No module named
  '_duckdb._sqltypes'`` even though ``import duckdb`` works standalone.
- Root cause: pytest-cov eagerly imports the coverage target; importing the
  ``pysharpe.validation`` package pulls ``duckdb`` through
  ``validation/ledger.py`` while coverage's import tracing is active, which
  trips duckdb's shim/extension handoff.
- Fix/workaround: always use the path form ``--cov=src/pysharpe`` (as in the
  Makefile); do not use ``--cov=pysharpe.<module>`` dotted targets.

### 2026 — PurgedKFold last fold bled into the previous fold's embargo

- Symptom: when the series length did not divide evenly into folds, the
  final test fold was clamped to the series end, overlapping or abutting
  the embargo zone that separates it from the previous test fold — leaking
  serially correlated information between adjacent test folds.
- Root cause: `test_end_idx >= n` clamping recalculated the fold boundaries
  without re-enforcing the embargo gap.
- Fix: fold size is computed exactly as
  `(n − (n_splits−1)·embargo) // n_splits`; trailing remainder observations
  are excluded rather than clamped.  Every adjacent test-fold pair is now
  separated by ≥ embargo observations by construction.
- Regression tests: `tests/test_resampling.py::TestPurgedKFoldLeakagePrevention`
- Grep guard: `grep -rn "test_end_idx >= n" src/pysharpe/validation/resampling.py`
  must be empty.

### 2026 — Walk-forward backtests silently ignored transaction costs

- Symptom: `WalkForwardBacktester` constructed `HistoricalBacktester`
  sub-windows with default (zero) cost parameters, so every window
  transition traded from old weights to newly optimized weights for free.
- Fix: `fee_per_trade`, `slippage_pct`, and `spread_pct` are now
  constructor parameters of `WalkForwardBacktester` and are forwarded to
  each sub-backtester.  The transition trade executes at the first close
  of the test window — costs never use future prices.
- Regression tests: `tests/test_analysis_transaction_costs.py::TestWalkForwardCosts`

### 2026 — DSR now deflates the observed Sharpe by the Lo θ factor

- Symptom: `compute_dsr` ignored return autocorrelation, so serially
  correlated strategies (e.g. smoothed/leveraged return streams) received
  inflated Deflated Sharpe Ratios.
- Fix: `compute_dsr(..., theta=1.0)` deflates the observed Sharpe by √θ
  (Lo 2002 Newey–West variance-inflation factor) before deflation;
  `compute_validation_metrics` computes θ from the return series
  automatically.  θ = 1 recovers the IID formula exactly.
- Regression tests: `tests/test_validation_metrics.py::TestDSRLoAdjustment`

### 2026 — Estimator validation now raises `DataValidationError`, not `ValueError`

- Symptom: callers catching `ValueError` around `compute_linear_shrinkage` /
  `compute_nonlinear_shrinkage` broke when structural validation (empty data,
  too few observations, non-numeric/infinite values, fully-missing assets)
  was migrated to `DataValidationError`.
- Root cause: `DataValidationError` subclasses `PySharpeError`, not
  `ValueError`, so legacy `except ValueError` blocks no longer intercept it.
- Fix: catch `DataValidationError` (or `PySharpeError`) at call sites.
  `TypeError` is still raised for non-DataFrame inputs.
- Guardrail: `grep -rn "except ValueError" src/pysharpe | grep -i -E "shrink|cov"`
  should be empty.

### 2026 — Zero-variance assets in HRP bisection

- Symptom: HRP crashed with `ZeroDivisionError` (or silently produced NaN
  weights) when a cluster contained a zero-variance asset whose covariance
  was exactly zero without triggering the correlation-ridge path.
- Root cause: `1.0 / np.diag(sub_cov)` divides by an exact zero when ridge
  regularisation did not fire (e.g. user-supplied covariance with a zero
  diagonal entry but finite off-diagonals).
- Fix: `_get_cluster_variance` floors each cluster variance at
  `max(max_var · 1e-12, 1e-15)` before inversion, and `_recursive_bisection`
  treats non-finite cluster variances as degenerate (equal split).
  User-supplied covariance matrices are additionally validated (finite,
  symmetric, PSD) up front via `DataValidationError`.
- Regression tests: `tests/test_optimization_hrp.py::TestZeroVarianceAssets`,
  `TestCovMatrixValidation`.

### 2026 — PyMC posterior covariance must be eigen-clipped before solvers

- Symptom: a near-singular posterior covariance (e.g. two nearly identical
  ETFs sampled by PyMC) crashed convex solvers, or silently produced
  nonsense when fed to `EfficientFrontier`.
- Root cause: the posterior covariance is the mean over MCMC draws of the
  LKJ-prior covariance — PSD in exact arithmetic, but numerically singular
  for collinear assets.
- Fix: `BayesianOptimizer.get_posterior_estimates()` symmetrises and
  eigen-clips the posterior covariance via `ensure_strictly_psd` (floor
  `max(λ_max · 1e-12, 1e-15)`).  `blend_views` applies the same hardening to
  its input and to Σ_p.  The `EfficientFrontier` integration
  (`BayesianOptimizer.optimize_efficient_frontier`) receives this hardened
  posterior covariance — never the raw sample covariance — and surfaces
  solver failures as `RuntimeError` instead of falling back to equal weights.
- Regression tests: `tests/test_optimization_bayesian.py::TestEfficientFrontierIntegration`,
  `tests/test_black_litterman.py::TestHardeningStrictPSD`.
- Grep guard: `grep -rn "sample" src/pysharpe/optimization/bayesian.py` must
  not appear in the frontier path.

### 2026 — Linear Ledoit-Wolf shrinkage is analytical, not sklearn

- Symptom: sklearn's `LedoitWolf` (previously wrapped by
  `compute_linear_shrinkage`) behaves opaquely on degenerate inputs —
  zero-variance columns and T < N regimes — and offered no strict-PSD
  guarantee.
- Fix: `compute_linear_shrinkage` now implements Ledoit & Wolf (2004)
  directly (π̂ / ρ̂ / γ̂ estimators, constant-correlation target, vectorised
  inner loop).  The diagonal exactly preserves sample variances; zero-
  variance assets are treated with zero correlation toward the target.
- Guardrail: `grep -rn "LedoitWolf" src/pysharpe/optimization/estimators.py`
  must be empty; sklearn may still be used elsewhere (pypfopt's
  `CovarianceShrinkage` in `portfolio_optimization.py`).

### 2026 — ruff `target-version` must match `requires-python`

- Symptom: raising ruff `target-version` from `py39` to `py312` (to match
  `requires-python = ">=3.12"`) surfaced ~20 findings (`B905`, `UP007`,
  `UP017`, `UP035`, `UP045`) in previously green files.
- Root cause: pyupgrade only suggests 3.10+/3.12 syntax once the target is
  raised; the codebase still carries pre-3.10 typing idioms.
- Fix: keep `target-version = "py312"` (correct metadata) and ignore those
  codes with a comment; remove the ignores when the code is modernized.
- Guardrail: new rule families (`C4`, `SIM`, `RET`, `RUF`, `PERF`, `TCH`,
  `PT`, `S`, `DTZ`, ...) all have dozens-to-hundreds of existing findings.
  Do not enable them without a dedicated cleanup phase.

### 2026 — `uv sync --locked` fails after any pyproject dependency edit

- Symptom: `uv lock --check` reports the lockfile needs updating after any
  `dependencies`/`optional-dependencies` change in `pyproject.toml`.
- Fix: run `uv lock && uv sync --all-extras` locally and commit `uv.lock` in
  the same change as the `pyproject.toml` edit.
- Guardrail: CI installs with `--locked`; never merge a dependency change
  without the regenerated lockfile.

### 2026 — mkdocs `--strict` aborts on griffe docstring warnings

- Symptom: `make build_docs` fails with "No type or annotation for returned
  value" / "No type or annotation for parameter '**kwargs'" / "Confusing
  indentation" warnings from griffe.
- Root cause: griffe's Google parser only accepts
  `name (type): description` (or a `\w+`-only type) in `Returns:` sections;
  dotted types like `matplotlib.axes.Axes: ...` fall through to the
  description and trigger the missing-annotation warning. `**kwargs:`
  without a parenthesized type fails likewise.
- Fix: write `axes (matplotlib.axes.Axes): ...` and `**kwargs (dict): ...`.
  Avoid bulleted lists with em-dash separators in `Returns:` sections.
- Guardrail: `mkdocs build --strict` is part of CI; any new docstring must
  parse cleanly under griffe.

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
