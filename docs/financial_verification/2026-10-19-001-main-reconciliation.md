# FMV-2026-10-17-001: main reconciliation, 2026-10-19

Continuation of [the missing-price audit](../financial_verification.md), delivered
through [draft PR #100](https://github.com/janusson/PySharpe/pull/100).
This is maintenance of the existing audit, not a new numerical finding.

## Trigger, revisions, and resolution

On 2026-10-19 (America/Vancouver), GitHub reported PR #100 open, draft, and
conflicting despite all eight checks passing on its unchanged previous head
`0ea78638488bc51d4c26ab4fc03e34c69c6e344b`. The default branch advanced from
`e4fb3f78f3046366036403c2ca33d9df1a7c83f7` to
`001f0c7d217f685d4707f14665ebe03fb5e84fe1` through merged PR #98. Main's CI
passed at that revision. The audit report/register are still absent from main;
PR #100 has not merged. PR #103 (historical Sharpe) and PR #104 (holdings across
walk-forward folds) remain separate open drafts; their latest quality checks
fail. Neither is superseded or integrated by this continuation.

The tested source/regression revision is merge commit
`b2bd4789633e55295bba10c5c21042c2c71bd9e1`; subsequent documentation-only commits
retain that code. Its parents are the two full head revisions above.

A normal merge of the pinned main revision into the owned audit branch preserves
history without rewriting either parent. Two mechanical conflicts were resolved:

- `docs/GOTCHAS.md`: preserve the missing-price entry and both new main entries
  for empty-series provenance and fixed-fee boundary NAV.
- `tests/test_categorization.py`: retain main's already-cleaned test, including
  its monkeypatched backward-fill tripwire; redundant local imports are removed.

Relative to main, production/test differences remain only the original
`metrics.py` correction and `test_metrics.py` regressions. Main's tactical changes
and public exports are retained unchanged. The original numerical correction,
independent scalar reference, sparse-frame regressions, and documented limits
are unchanged. This update introduces no additional financial semantics.

## Mathematics and retained independent evidence

For adjacent original finite price observations, simple returns satisfy
`r[t] = P[t] / P[t-1] - 1`; log returns satisfy
`l[t] = log(P[t] / P[t-1])`. A missing or non-finite endpoint leaves that interval
undefined. Undefined return rows may be removed only after differencing.
The retained scalar reference uses Python `math` and positional adjacency,
independently of pandas differencing and production input preparation. The
seeded sparse-frame tests cover four synthetic ETF series, partial/whole gaps,
simple/log methods, both drop policies, and duplicate labels (seed 20261017).
Original fail-before/pass-after evidence remains in the first report.

## Validation environment and commands

Python 3.12.15, pandas 3.0.3, NumPy 2.4.5; `uv sync --locked --all-extras`
passed against the committed lock. The imported module path was verified inside
`/Users/ericjanusson/.codex/worktrees/verify-missing-price-returns/PySharpe`.
The primary dirty checkout and other worktrees were preserved.

| Command/check | Result |
| --- | --- |
| `uv sync --locked --all-extras` | Passed; 130 packages audited |
| `uv run pytest tests/test_metrics.py tests/test_analysis_comparison.py tests/test_categorization.py tests/test_tactical_momentum.py` | 133 passed |
| `make check` | Passed: Ruff lint/format (122 files), Pyright 0 errors/warnings, 1,128 passed, 3 skipped, 9 warnings, 83.61% coverage (75% gate) |
| `make build_docs` | Strict build passed; repeated after adding this dated report |
| `git diff --check` | Passed after resolving conflicts |
| Independent continuation review | No actionable findings; 139 targeted tests passed and all 8 sparse-frame cases rejected the original baseline |
| Remote snapshot at `afeac2522b1d01a0804570d7e461bf565993de53` | All 8 checks passed, including installed wheel; source/test revision remains `b2bd478` |

The independent reviewer compared merge `b2bd478` against both parents, confirmed
that metrics code/oracle regressions match the audit parent and every other
production/test file matches main, and verified all gotcha sections survive.
Their targeted run covered metrics, comparison, categorization, package API, and
tactical tests. They did not independently rerun the full suite, docs, or CI.

An initial targeted invocation additionally named nonexistent
`tests/test_tactical_walk_forward.py`; pytest exited 4 before running tests.
The corrected command above covers walk-forward tests in
`tests/test_tactical_momentum.py` and passed. This failed invocation is not
counted as validation. No local installed-wheel rerun was performed; packaging
must be verified by the existing installed-wheel CI job on the pushed head.

Remote snapshot recorded 2026-10-19, 12:03 UTC: all eight GitHub checks passed
on documentation head `afeac2522b1d01a0804570d7e461bf565993de53`:
[PySharpe CI](https://github.com/janusson/PySharpe/actions/runs/39248845758),
[CodeQL](https://github.com/janusson/PySharpe/actions/runs/39248840396).
This includes the existing isolated installed-wheel smoke check. GitHub now
reports the PR mergeable; it remains open/draft with no submitted GitHub reviews.
This snapshot does not assert success for a later head; the evidence-only commit
retaining this snapshot must have its own actual-head checks inspected. Local
independent review is separate from a submitted GitHub approval. No full suite
was repeated for documentation-only edits after the tested source revision.

## Boundaries and remaining action

The original pattern search was refreshed with
`rg -n 'bfill\(|pct_change|dropna|shift' src/pysharpe/metrics.py src/pysharpe/analysis/comparison.py src/pysharpe/analysis/benchmarks.py`.
Caller-side price compaction remains in comparison and benchmark paths; this
helper cannot restore removed rows. Calendar-time annualization, absent dates,
nonpositive-price policy, and corporate-action validity remain outside the audit.
Passing regressions and CI do not certify financial results or empirical returns.

The continuation changes no allocator weights, MER/FX/tax rules, cache policy,
or public API beyond retaining already merged main changes. No new tests are
needed for a new defect because no new defect is corrected here. No new audit
or competing PR was opened. Human review and integration approval for #100,
including coordination with #103, remain required. Never infer integration from
a green PR head. No merge into main, auto-merge, release, brokerage access, or
trades were performed.
