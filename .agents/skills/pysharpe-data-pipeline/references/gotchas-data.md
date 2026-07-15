# Data Pipeline Gotchas

Failure patterns specific to the data ingestion layer.

---

## FX Lookahead Bias from bfill

- **Symptom**: Future exchange rates were applied to historical prices,
  inflating backtest returns with information that wasn't available at the time.
- **Root cause**: `apply_fx_conversion` called `.bfill()` after `.ffill()` on
  aligned exchange-rate series.
- **Fix**: Removed `.bfill()`. Leading rows without FX coverage are now detected
  and excluded with a warning. If all rows are excluded, a `ValueError` is raised.
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
- **Regression test**: `test_collation_service_uses_settings_cache_dir`
- **Grep guard**: `grep -rn 'DuckDBCachedPriceFetcher' src/pysharpe/ --include='*.py'`

## Pandas 2.2+ groupby(axis=1) Deprecation

- **Symptom**: Warning/crash in the collation layer on pandas ≥ 2.2.
- **Root cause**: `groupby(axis=1)` was deprecated in newer pandas versions.
- **Fix**: Replaced with transposed operations or column-wise iteration.
- **Regression test**: Covered by existing collation tests.
- **Grep guard**: `grep -rn 'groupby.*axis=1' src/pysharpe/ --include='*.py'`
