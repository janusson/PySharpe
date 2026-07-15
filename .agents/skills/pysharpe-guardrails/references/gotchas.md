# Known Failure Patterns (Gotchas)

Every entry represents a bug that shipped and was later fixed.
The goal is to prevent recurrence.

---

## MER Double-Division

- **Symptom**: Reported portfolio MER was 100× smaller than intended. An ETF
  with a 0.17% MER showed as 0.0017%.
- **Root cause**: MER values already stored as decimal fractions (0.0017) were
  divided by 100 a second time inside `SharpeOptimizer` and
  `optimise_portfolio_for_sharpe`.
- **Fix**: Removed the redundant `/ 100` division. Default VEQT MER corrected
  from `0.17` (percentage point) to `0.0017` (decimal fraction).
- **Regression test**: `test_sharpe_optimizer_mer_deduction_is_decimal_not_percentage`
- **Grep guard**: `grep -rn '/ 100' src/pysharpe/ --include='*.py' | grep -i mer`

## FX Lookahead Bias from bfill

- **Symptom**: Future exchange rates were applied to historical prices,
  inflating backtest returns with information that wasn't available at the time.
- **Root cause**: `apply_fx_conversion` called `.bfill()` after `.ffill()` on
  aligned exchange-rate series.
- **Fix**: Removed `.bfill()`. Leading rows without FX coverage are now
  detected and excluded with a warning. If all rows are excluded, a
  `ValueError` is raised.
- **Regression test**: `test_apply_fx_conversion_excludes_rows_with_no_fx_data`
- **Grep guard**: `grep -rn 'bfill()' src/pysharpe/ --include='*.py'`

## Stale LRU Cache on Collated CSVs

- **Symptom**: After re-downloading prices, the optimizer silently used the old
  collated data because the LRU cache key didn't include file modification time.
- **Root cause**: `_cached_collated_prices` was keyed on `(portfolio_name,
  collated_dir, time_constraint)` only.
- **Fix**: Added `csv_path.stat().st_mtime` as a fourth cache key, forcing a
  cache miss whenever the file is overwritten.
- **Regression test**: `test_load_collated_prices_reflects_updated_file`
- **Grep guard**: `grep -rn '@lru_cache' src/pysharpe/ --include='*.py'`

## DuckDB Wrapping Custom Fetchers

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

## Infeasible Geo Constraint on Missing Regions

- **Symptom**: `"infeasible solver"` crash when a portfolio had no assets mapped
  to a region with a lower-bound constraint.
- **Root cause**: Geographic lower-bound constraints were applied blindly to all
  configured regions, even those absent from the portfolio.
- **Fix**: The optimizer now drops lower-bound constraints for regions that
  contain no mapped assets.
- **Regression test**: `test_optimise_portfolio_respects_constraints` (in
  `test_portfolio_optimization.py`)
- **Grep guard**: `grep -rn 'lower_bound' src/pysharpe/ --include='*.py'`

## Pandas 2.2+ groupby(axis=1) Deprecation

- **Symptom**: Warning/crash in the collation layer on pandas ≥ 2.2.
- **Root cause**: `groupby(axis=1)` was deprecated in newer pandas versions.
- **Fix**: Replaced with transposed operations or column-wise iteration.
- **Regression test**: Covered by existing collation tests.
- **Grep guard**: `grep -rn 'groupby.*axis=1' src/pysharpe/ --include='*.py'`
