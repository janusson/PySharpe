# Financial verification: returns across missing prices

Audit **FMV-2026-10-17-001**, conducted 2026-10-17 (America/Vancouver).
See the [audit register and continuation protocol](financial_verification/index.md).

- Delivery: [PR #100](https://github.com/janusson/PySharpe/pull/100); its live
  status is authoritative. This report does not assert that it has merged.
- Original correction: `4e57e2df7bc926a0327420ac94bba03e292b4b3b`.
- Reference environment: Python 3.12.15, pandas 3.0.3, NumPy 2.4.5, and the
  committed `uv.lock`. The imported module was verified inside the dedicated
  `verify-missing-price-returns` worktree, not the dirty primary checkout.
- Reproduction: `uv sync --locked --all-extras`, then
  `uv run pytest tests/test_metrics.py tests/test_analysis_comparison.py`.
  Integration gates: `make check` and `make build_docs`.

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

### Original correction, revision `4e57e2d`

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
files. At this revision, repository-wide lint was blocked by inherited `I001` / `F841` findings
in `tests/test_categorization.py`; repository-wide formatting also flags that
file and `src/pysharpe/analysis/categorization.py`. Those files were unchanged in
the original correction, and the same findings were verified from the base commit.
The [initial CI run](https://github.com/janusson/PySharpe/actions/runs/39095143468)
passed tests, docs, installed-wheel smoke, and the secret scan; its quality job
failed on those inherited lint findings. This historical result is separate
from checks on later PR heads.

### Durability follow-up

The follow-up removes only the inherited import/unused-variable/formatting
blockers in the categorization files; their production behavior is unchanged.
It adds a seeded sparse-frame oracle test across both return methods and drop
policies, with and without duplicate index labels. The register, developer
entry point, documentation navigation, test map, and changelog make the invariant
and its reproduction discoverable beyond the original chat. These regressions
run in the normal CI suite.

### Continuation validation, 2026-10-18

The interrupted follow-up was recovered on the existing PR branch. Its tested
code and regression revision is `5c2bf4c072e89c6da2aa137ec7e8e62c50f261ae`;
the base remains `e4fb3f78f3046366036403c2ca33d9df1a7c83f7`. Subsequent report-only
commits do not change that tested code. Local validation used the reference
environment above in the same dedicated checkout, on 2026-10-18 at 08:19 PDT
(15:19 UTC):

| Command | Result |
| --- | --- |
| `uv run pytest tests/test_metrics.py tests/test_analysis_comparison.py tests/test_categorization.py` | 110 passed |
| `make check` | Ruff lint/format passed (121 files); Pyright: 0 errors, 0 warnings; 1,115 tests passed, 3 skipped, 9 warnings; coverage 83.51%, above the unchanged 75% gate |
| `make build_docs` | Strict build passed; the historical report remains linked from the navigable register |
| `git diff --check` | Passed |

The three skipped tests and nine warnings are retained as limits of the full
suite result; a passing gate does not mean every test executed. No local
installed-wheel rerun was performed: packaging is checked by the existing
installed-wheel CI job on the actual pushed PR head. Remote checks and integration
status must be read separately from these local results.

Remote snapshot, 2026-10-18 08:22 PDT: all eight checks passed on
`b941cc4d65461b65aed67c1258310d023aaf530d`, including quality, coverage,
strict docs, installed-wheel smoke, secret scanning, and CodeQL. The retained
[CI run](https://github.com/janusson/PySharpe/actions/runs/39140059416)
and [CodeQL run](https://github.com/janusson/PySharpe/actions/runs/39140056781)
identify that head; this snapshot does not assert success for later heads.
PR #100 was open and draft, with no submitted GitHub reviews or merge. Human
review and integration approval remain required; PR #103 also needs ordinary
overlap reconciliation. Later report-only commits preserve the tested code.

All eight new seeded sparse-frame cases also rejected the original base code.
The following offline command reuses the retained scalar oracle without changing
the working source or removing the correction:

```bash
uv run python - <<'PY'
import importlib.util
import subprocess
import types

spec = importlib.util.spec_from_file_location("verification_tests", "tests/test_metrics.py")
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
baseline = types.ModuleType("baseline_metrics")
source = subprocess.check_output(
    ["git", "show", "e4fb3f7:src/pysharpe/metrics.py"], text=True
)
exec(compile(source, "baseline_metrics.py", "exec"), baseline.__dict__)
tests.compute_returns = baseline.compute_returns
failures = 0
for method in ("simple", "log"):
    for dropna in (False, True):
        for duplicate_index in (False, True):
            try:
                tests.test_compute_returns_sparse_frame_matches_reference(
                    method, dropna, duplicate_index
                )
            except AssertionError:
                failures += 1
assert failures == 8, failures
print("All 8 seeded sparse-frame cases reject the original implementation.")
PY
```

The original independent review found no actionable findings and checked 400
randomized scalar-reference comparisons, duplicate indexes, and nullable numeric
gap recovery. A second independent review on 2026-10-18 found no actionable
findings in the continuation or original correction; its documented `uv run`
invocation passed all eight sparse-frame cases. A direct virtualenv invocation
initially failed in the existing pytest/Pyright hook before test collection; it
does not count as a successful test run. The reviewer did not run the full suite
or remote CI. The inherited categorization cleanup changes only formatting,
import order, and removal of an unused test binding. No allocator weights, fee
units, FX rules, tax fractions, caches, public exports, or financial semantics
beyond the original missing-price correction were changed. The pattern search
and unresolved caller-side compaction described below remain applicable.

PR #100 remains the delivery record; PR #103 separately corrects historical
Sharpe arithmetic and overlaps the metrics tests and documentation. Neither
open PR is evidence of integration. Reconcile both during human integration;
this continuation starts no competing audit or verification PR.

### Limits and follow-up candidates

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
