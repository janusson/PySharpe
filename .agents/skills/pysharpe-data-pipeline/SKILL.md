---
name: pysharpe-data-pipeline
description: >-
  PySharpe data ingestion pipeline: fetching, caching, FX conversion, and
  collation. Invoke when working with yfinance price downloads, DuckDB
  write-through caching (DuckDBCachedPriceFetcher), CAD/USD FX rate
  application, CSV collation of per-ticker price files, proxy_map.json
  ticker resolution, or data/portfolio CSV loading. CRITICAL GUARDRAILS:
  never .bfill() FX-aligned time series (lookahead bias), exclude rows
  without rate coverage instead of backfilling, DuckDB wrapping applies
  only to YFinancePriceFetcher (not custom/test stubs), collation cache
  keys MUST include csv_path.stat().st_mtime for invalidation.
---

# PySharpe Data Pipeline

Covers the full data ingestion stack: fetching prices, caching them in DuckDB,
converting currencies, collating per-ticker CSVs, and resolving proxy tickers.

## Module Locations

- `src/pysharpe/data/fetcher.py` — Price fetching and FX conversion.
- `src/pysharpe/data/collation.py` — CSV collation into unified DataFrames.
- `src/pysharpe/data/linkage.py` — Ticker chain linkage (e.g., delisted→replacement).
- `src/pysharpe/config.py` — `PySharpeSettings` dataclass and `get_settings()`.

## Architecture

### Price Fetcher Hierarchy

```
PriceFetcher (ABC)
├── YFinancePriceFetcher      — Live yfinance downloads.
└── DuckDBCachedPriceFetcher  — Write-through cache wrapping YFinancePriceFetcher.
```

- `YFinancePriceFetcher.fetch(ticker, start, end, base_currency)` returns a
  `pandas.DataFrame` with columns: `Date`, `Close`, `Currency`.
- If `base_currency` differs from the native currency, FX conversion is applied
  during fetch.
- **DuckDB wrapping is conditional**: only `YFinancePriceFetcher` instances get
  wrapped. Custom fetchers and test stubs pass through unwrapped.

### FX Conversion

`apply_fx_conversion(price_df, base_currency)`:
1. Fetches the CAD/USD (or relevant pair) exchange rate for the same date range.
2. Left-joins rates onto price data by date.
3. **Excludes rows without FX rate coverage** — emits a warning for partial
   coverage, raises `ValueError` if no rows remain.
4. **Never calls `.bfill()`** — forward-fill only (`.ffill()` is acceptable
   for intra-day alignment but NOT for historical backfill).

### Collation

`CollationService.collate(portfolio_name, fetcher, settings)`:
1. Reads the portfolio CSV from `data/portfolio/<name>.csv`.
2. Fetches each ticker via the provided fetcher.
3. Joins all ticker DataFrames on `Date` into a single collated DataFrame.
4. Writes the result to `data/exports/<name>_collated.csv`.

### Proxy Resolution

`proxy_map.json` in the working directory maps ticker → proxy ticker with
optional adjustments:
- `ticker` — The proxy ticker symbol.
- `fx` — FX rate override (float).
- `weight` — Position weight adjustment.

Loaded automatically by `build_settings()` at startup.

### Cache Invalidation

Collation is cached via `@lru_cache` with a key that includes
`csv_path.stat().st_mtime`. This forces a cache miss when the collated CSV is
overwritten.

## Critical Guardrails

Load `references/gotchas-data.md` for the full catalogue of data-pipeline
failure patterns with root causes, regression tests, and grep guards.

Quick reference:
- **No `.bfill()`** — `grep -rn 'bfill()' src/pysharpe/ --include='*.py'`
- **mtime in cache keys** — `grep -rn '@lru_cache' src/pysharpe/ --include='*.py'`
- **DuckDB wrapping scope** — `grep -rn 'DuckDBCachedPriceFetcher' src/pysharpe/ --include='*.py'`

## Testing

Relevant test files:
- `tests/test_fetcher.py`
- `tests/test_fx_adjustment.py`
- `tests/test_data_fetcher_cache.py`
- `tests/test_collation.py`
- `tests/test_collation_proxy.py`
- `tests/test_data_linkage.py`
- `tests/test_data_linkage_stitched.py`

Run: `uv run pytest tests/test_fetcher.py tests/test_fx_adjustment.py tests/test_data_fetcher_cache.py tests/test_collation.py`
