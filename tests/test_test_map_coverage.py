"""Guard against a stale test-to-module map.

``docs/TEST_MAP.md`` is the contract developers and agent skills follow to run
the right test subset. A test module that never gets mapped is invisible to that
workflow, so keep the map complete in both directions.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_MAP = REPO_ROOT / "docs" / "TEST_MAP.md"
TESTS_ROOT = REPO_ROOT / "tests"

_HEADING = re.compile(r"^## `([A-Za-z0-9_]+\.py)`", re.MULTILINE)


def _documented_modules() -> set[str]:
    return set(_HEADING.findall(TEST_MAP.read_text(encoding="utf-8")))


def _test_modules() -> set[str]:
    return {path.name for path in TESTS_ROOT.glob("test_*.py")}


def test_every_test_module_is_mapped() -> None:
    """Each ``tests/test_*.py`` module must appear in ``docs/TEST_MAP.md``."""
    missing = sorted(_test_modules() - _documented_modules())

    assert not missing, "Test modules missing from docs/TEST_MAP.md:\n" + "\n".join(
        f"tests/{name}" for name in missing
    )


def test_mapped_modules_exist() -> None:
    """Every module mapped in ``docs/TEST_MAP.md`` must exist under ``tests/``."""
    stale = sorted(_documented_modules() - _test_modules())

    assert not stale, (
        "docs/TEST_MAP.md maps test modules that do not exist:\n"
        + "\n".join(f"tests/{name}" for name in stale)
    )
