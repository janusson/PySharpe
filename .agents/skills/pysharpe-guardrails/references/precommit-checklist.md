# Pre-Commit Verification Checklist (Expanded)

Each item includes the concrete verification step and the rationale.

---

## 1. MER Values — Decimal Fractions, Not Percentage Points

**Check**: `grep -rn 'MER\|mer' src/pysharpe/ --include='*.py'` and verify every
value is < 0.10.

**Rationale**: An MER of 0.17% should be stored as `0.0017`, not `0.17`.
Storing as percentage points leads to double-division bugs (see gotchas).

**Common mistake**: Writing `mer = 0.17` when you mean 0.17% (should be `0.0017`).

---

## 2. No .bfill() on Time-Series Data

**Check**: `grep -rn 'bfill()' src/pysharpe/ --include='*.py'` must return zero results.

**Rationale**: Backfilling creates lookahead bias — future data leaks into
historical positions, inflating backtest performance.

---

## 3. FX Conversion Excludes Rows Without Rate Coverage

**Check**: Verify that `apply_fx_conversion` or equivalent raises `ValueError`
when all rows lack FX coverage, and emits warnings for partial coverage.

**Rationale**: Silent backfilling of FX rates was a shipped bug. Rows without
rate data must be excluded, not filled from future observations.

---

## 4. AssetTaxCharacteristics Income Fractions Sum to 1.0

**Check**: For every `AssetTaxCharacteristics` instance, verify
`eligible_dividend + foreign_income + capital_gains + return_of_capital + other_income == 1.0`
(within floating-point tolerance).

**Rationale**: Tax calculations depend on correct income attribution. Partial
fractions silently produce incorrect tax drag estimates.

---

## 5. Cache Invalidation — mtime Keys and cache_clear() in Tests

**Check**:
- Every `@lru_cache`-decorated function that reads files must include
  `os.path.getmtime(path)` in its cache key.
- Every test that modifies env vars or files that affect cached functions must
  call `the_function.cache_clear()`.

**Rationale**: Stale caches produce silently wrong results (see gotchas).

---

## 6. New Public Symbols in _EXPORT_MAP, TYPE_CHECKING, AND __all__

**Check**: For every new public function/class/dataclass added to the package:
- Add an entry to `_EXPORT_MAP` dict in `src/pysharpe/__init__.py`
- Add the import under the `TYPE_CHECKING` block
- Add the symbol name to the `__all__` list

**Rationale**: The package uses lazy `__getattr__` imports. Missing registration
means the symbol is inaccessible at runtime.

---

## 7. Synthetic Test Data Only, Fixed Seeds

**Check**: `grep -rn 'yfinance\|requests\.get\|urllib' tests/ --include='*.py'`
must return zero results (excluding test infrastructure for mocking).

**Rationale**: Tests must be deterministic and offline. Network-dependent tests
are flaky and slow.

---

## 8. No Predictive ML, Day-Trading, or Gamified UI

**Check**: Review every new file/module for:
- Price-prediction ML models (LSTM, transformers, etc.)
- Day-trading signal generators
- Gamification elements (badges, leaderboards, streaks)

**Rationale**: PySharpe is a rigorous analytical tool, not a trading bot or
consumer app. These features are explicitly out of scope.
