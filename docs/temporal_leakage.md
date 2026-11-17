# Historical data: temporal contract and audit

## Status and checkpoint

Working checkout: `main` at `4211047` with uncommitted changes; this is not a
claim of repository-wide no-lookahead compliance. PRs [#100](https://github.com/janusson/PySharpe/pull/100)
(missing-price returns), [#103](https://github.com/janusson/PySharpe/pull/103)
(historical Sharpe) and [#104](https://github.com/janusson/PySharpe/pull/104)
(walk-forward holdings) are separate open drafts, not integrated here. Confirm
their live state before integration. This change touches the same backtest file
as #104 and requires integration review.

The preceding implementation sessions reported that their new regressions failed
before the fixes and passed afterwards: dashboard and FX ordering cases, and
seven calendar cases. Their targeted suites reported 44, 43 and 34 passing
tests respectively. These are session reports, not hosted CI results. Combined local validation: `make check` passed (Ruff, Pyright, 1,125 passed,
3 skipped, 83.70% coverage); `make build_docs` passed. `git diff --check` passed.
Hosted CI and installed-wheel smoke have not been run for this working tree.

Targeted combined validation: `uv run pytest tests/test_fetcher.py
 tests/test_fx_adjustment.py tests/test_data_fetcher_cache.py tests/test_collation.py
 tests/test_data_linkage.py tests/test_app_streamlit.py tests/test_app_helpers.py
 tests/test_analysis_backtest_engine.py tests/test_analysis_transaction_costs.py`
 passed (121 tests). Strict docs and diff checks were repeated after documentation
 edits. These changes remain uncommitted on `main`; no PR has been created.

Next: independently review the calendar timing and FX assumptions, reconcile
with PR #104, and prepare a bounded draft PR only with publication authorization.

## Intended point-in-time contract

For an output labeled as available at time `t`, the inputs, universe, and all
parameters must be available by `t`. With those fixed, appending or changing
future observations must not change that output. Full-window summaries may
change when their window changes; retrospective analyses must not be presented
as historical decisions. Calendar dates alone do not establish intraday data
availability. Adjusted histories can be revised after corporate actions, so a
current adjusted download is not automatically a point-in-time vintage.

- A leading missing price is unknown, not zero and never filled from the future.
  An interior missing observation is not proof of a flat return or tradability.
  Explicit carry-forward may describe the last observed mark, but requires an
  age limit and must not be confused with an executable quote. A suspension or
  delisting cannot silently become an indefinitely tradable stale price.
- A return for adjacent covered observations is `P[t] / P[t-1] - 1`; a log
  return is `log(P[t] / P[t-1])`. Missing endpoints leave it undefined. Dates
  absent from the index need an explicit exchange calendar before they can be
  classified as non-trading days rather than data gaps. Asynchronous histories
  require a documented common calendar and availability policy, not future-based
  universe selection.
- Price-only closes omit distributions; adjusted total-return series incorporate
  provider adjustments and are not necessarily executable transaction prices.
  Do not mix bases, double count distributions or fees, or assume revised series
  existed in their final form historically.
- Convert with an FX observation actually available no later than the price's
  decision timestamp, subject to an explicit staleness limit. Exclude uncovered
  rows; never fill them from a later rate. Same calendar date is insufficient
  where market closing times differ. `apply_fx_conversion` and `HistoryLinker`
  enforce a default 7-calendar-day staleness cutoff (`max_fx_staleness_days=7`,
  opt-out via `None`), while exact intraday release timestamps remain out of scope.
- Entry is at an explicitly specified execution timestamp; existing holdings earn
  price movement through the execution close before that close's rebalance.
  Exit, unavailable execution prices, delisted holdings and costs need explicit
  policies: do not assume a sale at an unavailable mark. A decision using a
  closing price cannot earn the preceding close-to-close return as a new holding.
- Observations unavailable at decision time remain unknown. A later execution
  check may affect fills, but must not rewrite the earlier signal.

## Data-flow and earliest information dates

| Stage | Earliest information date and audit status |
| --- | --- |
| Provider download and DuckDB cache (`data/fetcher.py`) | The provider's observation/revision availability, not merely its index date. Cache keys omit provider configuration/adjustment basis; cached values are not historical vintages. |
| FX conversion (`data/fetcher.py`, `data/linkage.py`) | The later of the asset quote and matched FX availability. Forward-fill runs chronologically and excludes carried FX quotes older than `max_fx_staleness_days=7` calendar days by default (inclusive `end_date` requests prevent boundary truncation). Exact-date reindexing and intraday release timestamps remain subject to provider daily closes. |
| Proxy stitching (`data/linkage.py`) | Proxy returns require their endpoint quotes; stitched absolute levels additionally require the future handover scalar. Proxy requests can ignore caller bounds. USD proxies now multiply by `USDCAD=X` (`USD * CAD/USD -> CAD`, matching `apply_fx_conversion`). |
| Collation and cached CSV (`data/collation.py`, `portfolio_optimization.py`) | Inputs and any proxy handover must be available. Cache mtime invalidation does not prove observation freshness or vintage correctness. Forward-filled values are marks, not evidence of trading. |
| Price returns (`metrics.py`, `analysis/comparison.py`, `analysis/benchmarks.py`) | Both original adjacent endpoints. PR #100 corrects a gap-bridging defect in the helper but remains unmerged; upstream row deletion can still bridge gaps independently. |
| Backtests (`analysis/backtest_engine.py`) | A close-based rebalance uses that close and prior holdings. This change triggers on the first observed close in a new period, avoiding a future-dependent final-row trigger. Whole-frame column/row filtering still lets a later listing change prior windows; #104 separately addresses cross-window holdings. |
| Monthly research (`analysis/tactical_momentum.py`) | A completed period and its observable close; execution availability belongs to the later execution date. Partial final-month signals and next-session availability affecting prior signal records remain open. |
| Comparison and dashboard (`analysis/comparison.py`, `app/data.py`, `app/charts.py`) | Data through the selected evaluation end; optimized in-sample comparisons are retrospective, not OOS. Comparison currency and proxy date bounds need correction. The dashboard now fills only in chronological order but can still carry stale prices. |
| Derived statistics (`data/linkage.py`, `validation/resampling.py`) | A rolling statistic requires its trailing input window; `vol_hist` currently uses the full sample. Retrospective regime labels and non-chronological CV are not point-in-time decisions. Some first CV folds train on later observations; duplicate/unsorted indexes also need validation. |

## Guardrails and outstanding assumptions

Semantic regression tests, not grep alone, should compare outputs on identical
historical prefixes after future values and availability are changed. Cover
leading/interior gaps, staggered listings, FX coverage, suspensions, delistings,
unequal histories, and fold/period boundaries. Current tests cover only a subset;
passing them is not a global guarantee. `tests/test_fx_adjustment.py`,
`tests/test_data_linkage.py`, `tests/test_app_helpers.py`, and
`tests/test_analysis_backtest_engine.py` exercise this change; #100 and #104
contain further relevant tests on their respective branches.

Decisions still required: exchange calendars and time zones; quote and FX release
timestamps and acceptable staleness; historic data revisions and adjustment
vintages; eligible universe at each date; suspension/delisting valuation and exit;
calendar-close execution convention; interval and annualization semantics for
unequal coverage; and how to handle unknown or infeasible transactions. None of
these should be inferred from future observations. The bounded changes do not
certify backtest accuracy, live investability, or empirical performance.
