# Contributing to PySharpe

Thanks for helping to improve PySharpe! This project values reliability,
reproducibility, and approachable tooling. The guidelines below outline how to
get started.

## Development environment
1. Install PySharpe in editable mode with the locked development toolchain (`uv`):
   ```bash
   uv sync --locked --all-extras
   ```

## Running tests, type checks, and lint checks
- Execute the full pre-commit gate (`ruff` lint/format, `pyright`, and `pytest` with coverage):
  ```bash
  make check
  ```
- Or run individual targets mirroring CI:
  ```bash
  make lint       # uv run ruff check . && uv run ruff format --check .
  make typecheck  # uv run pyright --warnings
  make test       # uv run pytest with coverage floor
  ```

The CI workflow mirrors these commands and must succeed for every contribution.

## Making changes
- Keep numerical outputs and public APIs backward compatible unless the change
  is explicitly communicated.
- Add or update docstrings/examples when introducing new functionality.
- Update `CHANGELOG.md` with a short bullet describing the change under the
  appropriate version heading.

## Releases
1. Bump the version in `pyproject.toml` and update the changelog.
2. Tag the release (`git tag vX.Y.Z`) and push the tag (`git push --tags`).
3. Build distributions with `python -m build` and upload when ready.

We are not publishing automated releases yet, so keep release steps manual for
now. Thanks again for contributing!
