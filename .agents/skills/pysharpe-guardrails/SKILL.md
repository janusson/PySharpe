---
name: pysharpe-guardrails
description: >-
  HUMAN-ONLY pre-commit verification checklist for PySharpe changes. Use BEFORE
  declaring any code change complete. Covers MER-as-decimal-fractions (<0.10),
  no-.bfill()-on-timeseries, FX-row-exclusion, AssetTaxCharacteristics
  fractions sum-to-1.0, cache mtime-invalidation, __init__.py
  _EXPORT_MAP/TYPE_CHECKING/__all__ registration, synthetic-only test data,
  and prohibition of predictive ML/day-trading/gamified UI. Also references
  known failure patterns (gotchas) that must not be reintroduced.
disable-model-invocation: true
---

# PySharpe Pre-Commit Guardrails

**This skill is for human use only.** The model MUST NOT invoke this skill
autonomously. It encodes the mandatory verification checklist and known
failure patterns that must be reviewed before every change is declared complete.

## Verification Checklist

Before declaring any change "done", verify every item:

1. **MER values** — All are decimal fractions (< 0.10), never percentage points?
2. **Lookahead bias** — No `.bfill()` on time-series data anywhere?
3. **FX conversion** — Excludes rows without rate coverage instead of backfilling?
4. **Tax fractions** — `AssetTaxCharacteristics` income fractions sum to 1.0?
5. **Cache invalidation** — LRU keys include mtime; `cache_clear()` called in tests?
6. **Public API** — New symbols registered in `_EXPORT_MAP`, `TYPE_CHECKING`, AND `__all__` in `src/pysharpe/__init__.py`?
7. **Test isolation** — Tests use synthetic data only, with fixed seeds, no network calls?
8. **Scope boundaries** — No predictive ML, day-trading logic, or gamified UI introduced?

## Bug-Fix Discipline

When fixing a bug, follow this three-step process:

1. **Regression test** — Add a test that fails before the fix and passes after.
2. **Anti-pattern grep** — Search the entire codebase for the same failure pattern.
3. **Document** — Record the failure pattern in `docs/GOTCHAS.md`.

## Reference Documents

For the full catalogue of known failure patterns with root-cause analysis, grep
guards, and regression test names, load:

- `references/gotchas.md` — All documented failure patterns (MER double-division,
  FX lookahead bias, stale LRU cache, DuckDB test-stub bypass, infeasible geo
  constraints, pandas groupby deprecation).

For the expanded checklist with per-item rationale and examples, load:

- `references/precommit-checklist.md` — Expanded version of the 8-item checklist
  with concrete what-to-check steps for each item.
