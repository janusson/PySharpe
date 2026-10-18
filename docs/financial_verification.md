# Financial verification: returns across missing prices

## Scope and finding

This audit tested one invariant in `metrics.compute_returns` against base commit
`e4fb3f78f3046366036403c2ca33d9df1a7c83f7`, using synthetic prices only. A
periodic return must use adjacent rows in the original input. Missing prices
must not disappear before differencing, and adding another asset must not change
the original asset's returns on those rows.

The implementation violated this invariant for both simple and log returns.
`_prep_numeric` removed wholly missing rows before the calculation. A price gap
therefore disappeared for a Series or a wholly missing DataFrame row, but
remained when an unrelated asset had a price on the same row. The existing tests
covered complete price histories, but did not exercise this invariant.

## Mathematical reference

For ordered positive prices `P[t]`, a simple return is
`r[t] = P[t] / P[t-1] - 1`, and a log return is
`l[t] = log(P[t] / P[t-1])`. Both are undefined when either original endpoint is
missing or non-finite. The first row is also undefined. Removing undefined
returns afterward does not create an observation for a missing interval.

The independent test oracle uses a scalar loop over the original adjacent
endpoint pairs, `math.isfinite`, division, and `math.log`. It does not use the
production numeric preparation, pandas shifting, or `pct_change`.

For five consecutive business-day observations:

| Row | Price | Reference simple return | Original Series result |
| --- | --- | --- | --- |
| 0 | 100 | undefined | undefined |
| 1 | missing | undefined | row removed |
| 2 | 110 | undefined | +10% |
| 3 | 99 | -10% | -10% |
| 4 | 108.9 | +10% | +10% |

The log-return reference is
`[NaN, NaN, NaN, -0.10536051565782628, 0.09531017980432493]`.
Adding a constant, fully observed second asset made the original implementation
produce the correct reference for the first asset, exposing column dependence.

The invented observation affects downstream statistics. For the toy simple
returns above, the sample standard deviation annualized by `sqrt(252)` is
`1.8330302779823366` for the original output versus `2.2449944320643658` for the
two covered adjacent returns: an 18.35% understatement. The finite observation
count also changes from two to three, contaminating geometric annualization.
These short, deliberately volatile data demonstrate the numerical error; they
are not investment forecasts or adequate samples for performance estimation.

## Correction and compatibility

- Numeric preparation can retain wholly missing rows. Only `compute_returns`
  opts into this; the other metrics retain their cleaning default.
- Simple returns explicitly use `pct_change(fill_method=None)`, pinning the
  no-imputation policy across supported pandas versions. The reproduced defect
  occurred with pandas 3.0.3, whose default already avoids forward filling.
- `dropna=False` retains the complete original index. With `dropna=True`, wholly
  undefined return rows are removed after the adjacent-row calculation; partial
  missing values in a multi-asset row remain missing.
- Complete price histories retain the same return formulas. Incomplete histories
  now have fewer finite observations instead of invented multi-period moves.
  Downstream rolling volatility waits for a fully covered window after a gap.
  An input with no finite prices still raises `ValueError`; finite prices with
  no covered adjacent pair produce an empty result when `dropna=True`.

## Evidence and limits

The original metrics/comparison subset passed all 74 tests. The new regression
cases then produced 21 expected failures on the original code, while two
existing-validation checks passed. After correction, the expanded subset passed
all 97 tests. Coverage includes simple/log methods, Series/single-column frames,
leading/interior/consecutive/trailing gaps, non-finite and nonnumeric prices,
column independence, and rolling-volatility recovery. Tests use deterministic
synthetic values and make no market-data requests.

The full suite passed **1,107 tests**, with **3 skipped** and **83.51%** total
coverage (75% required). Type checking reported zero errors/warnings, the strict
documentation build passed, and lint/format checks passed for the changed Python
files. Repository-wide lint remains blocked by inherited `I001` / `F841` findings
in `tests/test_categorization.py`; repository-wide formatting also flags that
file and `src/pysharpe/analysis/categorization.py`. These files are unchanged by
this correction, and the same findings were verified from the base commit.

The pattern search covered `dropna`, `pct_change`, and `shift` throughout
`src/pysharpe`. Caller-side price compaction remains in `analysis/comparison.py`
and `analysis/benchmarks.py`; other direct `pct_change` calls also retain their
own policies. Those paths were inspected for scope, not corrected or certified
by this audit. This helper cannot reconstruct rows removed before it is called.

The invariant is about input-row adjacency. It does not infer an exchange
calendar, detect dates absent from the input index, set a stale-price tolerance,
or establish the validity of corporate-action adjustments. Annualization still
uses the caller's stated frequency and finite return count; it does not become
a calendar-time estimator for an incomplete historical period. Nonpositive
price policy and unrelated metrics are outside this correction. No brokerage
accounts, real holdings, or trades were accessed.
