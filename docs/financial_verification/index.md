# Financial correctness verification

This register preserves the mathematical question, independent reference,
regression tests, and delivery link for each bounded audit. It is the starting
point for both contributors and recurring verification runs. Software regression
evidence is not independent financial validation or investment advice.

## Audit register

| Audit ID | Invariant and outcome | Evidence | Regression coverage | Delivery |
| --- | --- | --- | --- | --- |
| FMV-2026-10-17-001 | Returns use adjacent original price rows; a gap-bridging defect was reproduced and corrected on the PR branch. | [Missing-price returns](../financial_verification.md), [2026-10-19 reconciliation](2026-10-19-001-main-reconciliation.md), original base `e4fb3f7` | `tests/test_metrics.py`: price gaps, column independence, sparse-frame oracle, and rolling-volatility recovery | [PR #100](https://github.com/janusson/PySharpe/pull/100) |

The audit outcome and delivery state are separate. The linked PR is authoritative
for its current head, checks, reviews, and merge state. An open PR is not an
integrated fix. A closed, unmerged PR is not a merged fix. Do not copy a historical
green check onto a newer head revision.

## Reproduce and keep the checks running

From a dedicated checkout of the PR branch or a revision containing the fix:

```bash
uv sync --locked --all-extras
uv run pytest tests/test_metrics.py tests/test_analysis_comparison.py tests/test_categorization.py
make check
make build_docs
```

The targeted command covers the first audit and its integration cleanup; use
[the test map](../TEST_MAP.md) for later audits. `make check` runs repository-wide
lint, type checking, and the full test suite with its coverage gate. The ordinary
[CI workflow](https://github.com/janusson/PySharpe/blob/main/.github/workflows/ci.yml)
runs those gates, strict documentation builds, and an installed-wheel smoke check
on PR updates and main. The oracle regressions belong to the normal test suite;
there is no separate opt-in verification runner to forget.

Tests use deterministic synthetic prices and fixed seeds. Missing-price tests
compare production output to scalar mathematics independent of production
preparation and pandas differencing, including sparse multi-asset frames and
duplicate row labels. Preserve the original reproduction and existing tests when
adding coverage; a changed expected result requires a documented financial reason.

## Continue prior work before starting another audit

1. Inspect the current default branch, this register, relevant open/closed PRs,
   worktrees, and reviews. Pending reports can exist only on a PR branch before
   merging; their absence from main does not mean the work has not been done.
2. Reconcile a pending verification PR first. Check its latest head, CI, review
   findings, conflicts, and whether it is already merged or superseded. Continue
   the existing branch when safe. Do not create a second PR for the same finding,
   rewrite collaborators' history, or overwrite a dirty checkout.
3. Fix relevant failures with the smallest correction and update its evidence.
   Scope inherited quality cleanup explicitly; do not weaken a gate or replace
   expected mathematical behavior merely to get a green result. Do not rerun
   expensive checks or rewrite reports when the code, base, and check evidence
   have not changed and there is no unresolved concern.
4. If an existing verification PR only awaits human approval, report the exact
   decision and retain it instead of building a backlog of new verification PRs.
   Never merge, enable auto-merge, or release as part of the recurring mission.
5. After GitHub confirms integration, verify that the regression tests and report
   are present on the current default branch, then select one new invariant from
   unresolved evidence. A closed/unmerged or superseded PR needs an explicit
   disposition before being skipped.

The current scheduled mission is a local Codex configuration, not repository
infrastructure. Its prompt points here and seeds PR #100 so it can find this work
before this register reaches main. Cloning the repository provides the tests and
CI configuration; it does not install a local scheduled task.

## Record a later audit

Add a stable ID and register row. Store new reports under this directory using
`YYYY-MM-DD-<invariant>.md`; retain the first report's existing URL. Each report
must include:

- The consequential invariant, scope, expected equations, and implications of
  any change to established financial semantics.
- Base revision, date/time zone, Python and relevant dependency versions, and
  tested source location. Identify the tested revision or retained CI run.
- A deterministic independent reference, a minimal reproduction, and evidence
  that the regression fails before a correction and passes after it.
- Exact validation commands, results, skipped checks, remaining uncertainty,
  related anti-pattern searches, and follow-up candidates. A no-defect result
  must state the tested boundary and limits instead of implying certification.
- The PR/delivery link and current blockers when applicable. Record material
  evidence changes with their revision; preserve earlier results as historical.

Update the changelog, [test map](../TEST_MAP.md), and
[gotchas](../GOTCHAS.md) when the finding changes behavior, test ownership, or
known failure patterns. Keep the essential evidence in Git: CI artifacts have
limited retention, and a chat or a successful local command is not the durable
record. No brokerage access, real holdings, trades, or predictive trading models
are needed for this workflow.
