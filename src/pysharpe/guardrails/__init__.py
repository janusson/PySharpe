"""Canadian tax compliance guardrails for multi-account portfolios.

Prevents automated rebalancing from triggering CRA Superficial Loss rules
and tracks Adjusted Cost Base (ACB) with commission support.
"""

from .tax_compliance import (
    ACBPosition,
    ACBTracker,
    SuperficialLossGuardrail,
    SuperficialLossViolation,
    TransactionRecord,
    build_default_identical_map,
)

__all__ = [
    "ACBPosition",
    "ACBTracker",
    "SuperficialLossGuardrail",
    "SuperficialLossViolation",
    "TransactionRecord",
    "build_default_identical_map",
]
