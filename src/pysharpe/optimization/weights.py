"""Helpers for post-processing portfolio allocations."""

from __future__ import annotations

import math
from typing import Mapping

ATOL: float = 1e-12
RTOL: float = 1e-9


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Return a copy of *weights* scaled to sum to 1.0.

    Args:
        weights: Mapping of asset identifiers to raw allocation weights.

    Returns:
        New dictionary whose values are non-negative, with negatives clipped to
        zero, and add up to one.

    Raises:
        ValueError: If the input contains NaNs/Infs or no positive weights.
    """

    cleaned: list[tuple[str, float]] = []
    total_weight = 0.0
    for key, raw in weights.items():
        weight = float(raw)
        if math.isnan(weight) or math.isinf(weight):
            raise ValueError(f"Non-finite weight encountered for {key!r}.")
        if weight <= 0.0:
            weight = 0.0
        cleaned.append((key, weight))
        total_weight += weight

    if total_weight <= ATOL:
        raise ValueError("At least one weight must be positive.")

    normalised: dict[str, float] = {}
    for key, weight in cleaned:
        normalised[key] = 0.0 if weight <= 0.0 else weight / total_weight

    # Enforce the sum invariant within tolerances by nudging the largest weight.
    total = math.fsum(normalised.values())
    if not math.isclose(total, 1.0, rel_tol=RTOL, abs_tol=ATOL):
        residual = 1.0 - total
        if residual:
            largest_key = max(normalised, key=normalised.__getitem__)
            normalised[largest_key] += residual

    final_total = math.fsum(normalised.values())
    if not math.isclose(final_total, 1.0, rel_tol=RTOL, abs_tol=ATOL):
        raise ValueError("Unable to normalise weights to sum to 1.0 within tolerance.")

    return normalised


__all__ = ["normalize_weights"]
