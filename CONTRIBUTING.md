# Contributing to PySharpe

Thanks for helping to make PySharpe better! This project values reliability,
reproducibility, and approachable tooling. The guidelines below outline how to
get started.

## Development environment
1. Create and activate a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install PySharpe in editable mode with development extras.
   ```bash
   pip install -e .[dev]
   ```

## Running tests and lint checks
- Execute the full test suite with coverage:
  ```bash
  python -m pytest
  ```
- Run the linters before opening a pull request:
  ```bash
  ruff check src tests
  ruff format --check src tests
  black --check src tests
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
