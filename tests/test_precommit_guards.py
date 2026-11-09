"""Mechanical guards for the ``AGENTS.md`` section 4 pre-commit checklist.

Section 4 asks an author to verify eight things before declaring a change
complete.  Four of them can be settled by a machine; this module settles the
two that had no test, and the other two are already enforced by
``AssetTaxCharacteristics.__post_init__`` and covered in
``tests/test_tax_location.py``.

Keeping a rule here instead of in the document means it runs on every
``make test``, CI included, rather than only when someone remembers.  The
remaining items need judgement and stay in ``AGENTS.md``: FX row exclusion
(behavioural, see ``tests/test_fx_adjustment.py``), cache invalidation
(per-caller, see ``docs/GOTCHAS.md``), synthetic data with no network calls,
and the no-predictive-ML scope rule.
"""

from __future__ import annotations

import importlib
import json
from collections import Counter
from pathlib import Path

import pytest

import pysharpe
from pysharpe.optimization.tax_location import AssetTaxCharacteristics

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "pysharpe"
EXAMPLE_CONFIG = REPO_ROOT / "portfolio_config.example.json"

# A backfilled price or FX series invents values the model could not have known
# at the time, so the pattern is kept out of the package wholesale rather than
# reviewed case by case.
FORWARD_FILL = ".bfill("


def _shipped_mer_values() -> dict[str, float]:
    """Return every MER the example portfolio config ships with."""
    config = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    values = {str(k): float(v) for k, v in config["mer_mapping"].items()}
    values["<constraints.max_portfolio_mer>"] = float(
        config["constraints"]["max_portfolio_mer"]
    )
    return values


# ---------------------------------------------------------------------------
# Item 6: public symbols are registered in _EXPORT_MAP and __all__
# ---------------------------------------------------------------------------


#: Snapshot so the parametrisation is stable for the whole session.
_EXPORTED_SYMBOLS = sorted(pysharpe._EXPORT_MAP)


@pytest.mark.parametrize("symbol", _EXPORTED_SYMBOLS)
def test_export_map_entry_resolves_and_is_advertised(symbol: str) -> None:
    """Each lazy export must resolve from its module and appear in ``__all__``.

    ``_EXPORT_MAP`` is the single declaration of the public surface; ``__all__``
    is derived from it, so an entry that cannot resolve means the map points at
    a stale module or attribute name.
    """
    module_name, attribute = pysharpe._EXPORT_MAP[symbol]
    module = importlib.import_module(module_name)

    assert hasattr(module, attribute), f"{module_name}.{attribute} does not exist"
    assert getattr(pysharpe, symbol) is getattr(module, attribute)
    assert symbol in pysharpe.__all__, f"{symbol} is missing from __all__"


def test_every_public_name_resolves() -> None:
    """Every name ``__all__`` advertises must be importable from the package."""
    unresolved = sorted(
        name for name in pysharpe.__all__ if not hasattr(pysharpe, name)
    )

    assert not unresolved, f"__all__ advertises unresolvable names: {unresolved}"


def test_all_has_no_duplicate_entries() -> None:
    """A name listed twice would make the public surface ambiguous."""
    duplicates = sorted(
        name for name, count in Counter(pysharpe.__all__).items() if count > 1
    )

    assert not duplicates, f"__all__ lists these names twice: {duplicates}"


# ---------------------------------------------------------------------------
# Item 2: no backfill on time-series data
# ---------------------------------------------------------------------------


def test_no_forward_fill_in_package_sources() -> None:
    """No module may backfill a time series; missing rows are dropped instead."""
    offenders = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in PACKAGE_ROOT.rglob("*.py")
        if FORWARD_FILL in path.read_text(encoding="utf-8")
    )

    assert not offenders, (
        "Backfilling a time series looks ahead. Drop the uncovered rows"
        f" instead: {offenders}"
    )


# ---------------------------------------------------------------------------
# Item 1: MERs ship as decimal fractions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ticker,mer", sorted(_shipped_mer_values().items()))
def test_shipped_mer_passes_the_runtime_validator(ticker: str, mer: float) -> None:
    """Every MER in the example config must satisfy ``AssetTaxCharacteristics``.

    The percent-points-instead-of-fraction mistake is silent until an optimiser
    deducts 100x the intended drag, so the shipped data is checked against the
    same validator the runtime uses rather than against a copy of the rule.
    """
    AssetTaxCharacteristics(ticker=ticker, dividend_yield=0.0, mer=mer)


def test_example_config_declares_mers() -> None:
    """Guard the parametrisation above against silently checking nothing."""
    assert _shipped_mer_values(), f"{EXAMPLE_CONFIG.name} declares no MER values"
