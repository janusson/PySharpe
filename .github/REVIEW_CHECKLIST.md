# Reviewer / acceptance checklist

Shared list for the author, the Zed reviewer, and the owner accepting a branch.
Nothing here replaces the domain rules in `AGENTS.md` or the pre-commit
verification list in its section 4.

## Author, before handoff

- [ ] The branch contains one concern; unrelated working-tree changes are
      excluded from `git diff --stat <base>`.
- [ ] The base `origin/main` SHA is recorded in the pull request.
- [ ] `make lint`, `make typecheck`, `make test`, `make build_docs` pass; the
      installed-wheel smoke has been run when public exports changed.
- [ ] New public symbols are registered in `_EXPORT_MAP`, `TYPE_CHECKING`, and
      `__all__` (`src/pysharpe/__init__.py`).
- [ ] `CHANGELOG.md`, `docs/TEST_MAP.md`, and `docs/GOTCHAS.md` are updated when
      the change warrants it.
- [ ] The `AGENTS.md` section 4 pre-commit list is answered explicitly.

## Reviewer, in Zed

- [ ] The diff matches the stated scope; nothing unrelated rides along.
- [ ] Financial semantics preserved: units and conventions (MER as a decimal
      fraction, ACB vs book cost, CAD base currency, no `.bfill()` on time
      series) are unchanged unless the change is about them.
- [ ] Boundary behaviour is tested: zero and negative quantities, non-finite
      prices, missing FX rate coverage, empty journals.
- [ ] Tests use synthetic data with fixed seeds and make no network calls.
- [ ] No tax-loss harvesting, no Markowitz substitution for the VA allocator, no
      single-stock modelling (`AGENTS.md` sections 3 and 6).
- [ ] Failure modes are loud: unknown prices and unresolvable currencies raise
      or are reported, never silently valued at zero.

## Owner, acceptance

- [ ] The stated verification re-runs green on the recorded base revision.
- [ ] Caller-visible API changes are described and intentional.
- [ ] Deferred items are recorded as follow-ups rather than dropped.
- [ ] Landing is an explicit decision; passing CI or a completed review does not
      imply authorisation to merge.
