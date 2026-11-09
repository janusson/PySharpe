## What changed

<!-- One paragraph. Name the modules and public symbols a reviewer should open first. -->

## Why

<!-- The problem this solves. Link the issue if there is one. -->

## Base revision

<!-- `git rev-parse origin/main` at the time the branch was cut. -->

- Base: `<sha>`
- Branch: `<branch>`

## Verification

<!-- Tick only what actually ran. Name anything skipped and why. -->

- [ ] `make lint`
- [ ] `make typecheck`
- [ ] `make test` (coverage gate)
- [ ] `make build_docs`
- [ ] Installed-wheel smoke (`scripts/smoke_wheel.py`)

Targeted subset run first per `docs/TEST_MAP.md`:

```
<command>
```

## Scope

<!-- What this branch deliberately does NOT change. Unrelated work carried in the
     same working tree must be listed here as excluded. -->

## Review notes

<!-- Domain decisions a reviewer should challenge: units, tax conventions,
     numerical tolerances, FX handling, public API changes. -->

See `.github/REVIEW_CHECKLIST.md` for the shared acceptance list.
