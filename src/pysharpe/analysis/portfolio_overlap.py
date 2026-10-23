"""Look-through overlap from caller-supplied holdings and constituents.

This module does not fetch, infer, or persist portfolio data. Constituents are
provided as fractions of each fund's value. Uncovered fund weight remains
unknown and is never redistributed across the supplied constituents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..exceptions import DataValidationError


@dataclass(frozen=True)
class PortfolioOverlapResult:
    """Deterministic portfolio exposures and pairwise shared weights."""

    portfolios: dict[str, dict[str, Any]]
    pairwise: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-compatible overlap contract."""
        return {
            "schema_version": "1.0",
            "analysis": "portfolio_overlap",
            "portfolios": self.portfolios,
            "pairwise": list(self.pairwise),
        }

    def to_json(self) -> str:
        """Serialize deterministically with standard JSON values only."""
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )


def analyze_portfolio_overlap(
    portfolios: Mapping[str, Mapping[str, float]],
    *,
    fund_constituents: Mapping[str, Mapping[str, float]],
    fund_tickers: set[str] | frozenset[str] = frozenset(),
) -> PortfolioOverlapResult:
    """Measure shared underlying weight using explicitly supplied look-through.

    ``portfolios`` maps portfolio names to holding weights. ``fund_tickers``
    identifies fund holdings for which a constituent map is expected; missing
    look-through for those holdings is counted as unknown. Non-fund holdings
    without a constituent map are treated as directly held securities.

    Constituent weights are fractions of fund value and may sum to less than
    one. The residual is reported as unknown. For each portfolio pair,
    ``shared_weight`` is the sum of minimum weights in each common underlying
    security; ``overlap_of_smaller_known_portfolio`` divides that amount by
    the smaller known exposure total.
    """
    if not portfolios:
        raise DataValidationError("at least one portfolio is required")
    _validate_constituents(fund_constituents)

    exposures: dict[str, dict[str, float]] = {}
    portfolio_rows: dict[str, dict[str, Any]] = {}
    for name, holdings in sorted(portfolios.items()):
        if not isinstance(name, str) or not name.strip():
            raise DataValidationError("portfolio names must be non-empty strings")
        if not holdings:
            raise DataValidationError(f"portfolio {name} has no holdings")
        total_weight = 0.0
        unknown_weight = 0.0
        known: dict[str, float] = {}
        for ticker, raw_weight in sorted(holdings.items()):
            if not isinstance(ticker, str) or not ticker.strip():
                raise DataValidationError("holding symbols must be non-empty strings")
            weight = _fraction(raw_weight, f"weight for {name}/{ticker}")
            total_weight += weight
            if ticker in fund_constituents:
                constituent_weights = fund_constituents[ticker]
                covered = sum(float(value) for value in constituent_weights.values())
                unknown_weight += weight * max(0.0, 1.0 - covered)
                for constituent, fraction in constituent_weights.items():
                    known[constituent] = known.get(constituent, 0.0) + weight * float(
                        fraction
                    )
            elif ticker in fund_tickers:
                unknown_weight += weight
            else:
                known[ticker] = known.get(ticker, 0.0) + weight
        if total_weight > 1.0 + 1e-12:
            raise DataValidationError(
                f"weights for portfolio {name} sum to more than 1.0"
            )
        known_total = sum(known.values())
        exposures[name] = known
        portfolio_rows[name] = {
            "input_weight": total_weight,
            "unallocated_weight": max(0.0, 1.0 - total_weight),
            "known_exposure": dict(sorted(known.items())),
            "known_weight": known_total,
            "unknown_weight": unknown_weight,
            "coverage": known_total / total_weight if total_weight else None,
        }

    pairwise: list[dict[str, Any]] = []
    names = sorted(exposures)
    for left_pos, left_name in enumerate(names):
        for right_name in names[left_pos + 1 :]:
            left, right = exposures[left_name], exposures[right_name]
            shared = {
                ticker: min(left[ticker], right[ticker])
                for ticker in sorted(set(left) & set(right))
            }
            shared_weight = sum(shared.values())
            smaller_known = min(sum(left.values()), sum(right.values()))
            pairwise.append(
                {
                    "portfolio_a": left_name,
                    "portfolio_b": right_name,
                    "shared_weight": shared_weight,
                    "overlap_of_smaller_known_portfolio": (
                        shared_weight / smaller_known if smaller_known > 0 else None
                    ),
                    "shared_securities": shared,
                }
            )
    return PortfolioOverlapResult(portfolios=portfolio_rows, pairwise=tuple(pairwise))


def _validate_constituents(
    fund_constituents: Mapping[str, Mapping[str, float]],
) -> None:
    for fund, constituents in fund_constituents.items():
        if not isinstance(fund, str) or not fund.strip() or not constituents:
            raise DataValidationError("fund constituent maps need a fund and rows")
        total = 0.0
        for ticker, raw_weight in constituents.items():
            if not isinstance(ticker, str) or not ticker.strip():
                raise DataValidationError(
                    "constituent symbols must be non-empty strings"
                )
            total += _fraction(raw_weight, f"constituent weight for {fund}/{ticker}")
        if total > 1.0 + 1e-12:
            raise DataValidationError(
                f"constituent weights for {fund} sum to more than 1.0"
            )


def _fraction(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise DataValidationError(f"{label} must be a finite fraction")
    fraction = float(value)
    if not np.isfinite(fraction) or fraction < 0.0:
        raise DataValidationError(f"{label} must be a finite non-negative fraction")
    return fraction


__all__ = ["PortfolioOverlapResult", "analyze_portfolio_overlap"]
