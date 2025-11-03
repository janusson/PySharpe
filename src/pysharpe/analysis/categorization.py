"""Tools for grouping assets into broader economic categories."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

import pandas as pd

from pysharpe.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CategoryAggregation:
    """Result of collapsing individual tickers into category proxies.

    Attributes:
        prices: Normalised price history with one column per category.
        category_lookup: Mapping of ticker -> resolved category label.
        groups: Mapping of category -> tuple of tickers included in the aggregate.
        dropped: Tuple of tickers excluded because they lacked a category.
    """

    prices: pd.DataFrame
    category_lookup: dict[str, str]
    groups: dict[str, tuple[str, ...]]
    dropped: tuple[str, ...]


def _normalise_mapping(mapping: Mapping[str, str]) -> dict[str, str]:
    normalised: dict[str, str] = {}
    for key, value in mapping.items():
        if not key:
            continue
        normalised[str(key).upper()] = str(value).strip()
    return normalised


def load_category_map(path: Path | str | None = None) -> dict[str, str]:
    """Load a ticker-to-category mapping from disk.

    The loader expects a JSON document structured as ``{"TICKER": "Category"}``.
    When *path* is omitted the function looks for ``asset_categories.json`` in
    the configured ``info_dir``. An empty mapping is returned when the file is
    missing so callers can treat categorisation as optional.
    """

    settings = get_settings()
    candidate = Path(path or settings.info_dir / "asset_categories.json").expanduser()
    if not candidate.exists():
        logger.debug("Category mapping not found at %s", candidate)
        return {}

    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unable to parse category mapping: {candidate}") from exc

    if not isinstance(payload, MutableMapping):
        raise ValueError(f"Category mapping must be an object: {candidate}")

    mapping: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Category mapping keys and values must be strings.")
        mapping[key] = value
    return mapping


def apply_category_mapping(
    prices: pd.DataFrame,
    mapping: Mapping[str, str],
    *,
    include_unmapped: bool = True,
) -> CategoryAggregation:
    """Collapse *prices* into category representatives.

    Args:
        prices: Price DataFrame indexed by date with one column per ticker.
        mapping: Ticker -> category mapping (case-insensitive on ticker).
        include_unmapped: When ``True`` retain tickers that do not have an
            explicit category by treating the ticker symbol as its own category.

    Returns:
        :class:`CategoryAggregation` containing the transformed price data and
        bookkeeping information about which tickers were grouped or dropped.

    Raises:
        ValueError: If no tickers remain after applying the mapping.
    """

    if prices.empty:
        return CategoryAggregation(
            prices=prices.copy(),
            category_lookup={},
            groups={},
            dropped=tuple(),
        )

    normalised_map = _normalise_mapping(mapping)
    frame = prices.copy()
    frame = frame.sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.ffill().bfill()

    column_order: list[str] = []
    category_labels: list[str] = []
    groups: dict[str, list[str]] = {}
    lookup: dict[str, str] = {}
    dropped: list[str] = []

    for column in frame.columns:
        ticker = str(column)
        category = normalised_map.get(ticker.upper())
        if category is None:
            if include_unmapped:
                category = ticker
            else:
                dropped.append(ticker)
                continue

        column_order.append(ticker)
        category_labels.append(category)
        groups.setdefault(category, []).append(ticker)
        lookup[ticker] = category

    if not column_order:
        raise ValueError("No tickers remain after applying category mapping.")

    frame = frame.loc[:, column_order]

    baseline = frame.iloc[0].replace(0, pd.NA)
    normalised = frame.divide(baseline)
    normalised = normalised.ffill().bfill()
    normalised.iloc[0] = 1.0

    ordered_categories = list(dict.fromkeys(category_labels))
    category_index = pd.Index(category_labels, name="Category")
    grouped = normalised.groupby(category_index, axis=1).mean()
    grouped = grouped.reindex(columns=ordered_categories)

    groups_readonly = {key: tuple(value) for key, value in groups.items()}

    return CategoryAggregation(
        prices=grouped,
        category_lookup=lookup,
        groups=groups_readonly,
        dropped=tuple(dropped),
    )


__all__ = [
    "CategoryAggregation",
    "apply_category_mapping",
    "load_category_map",
]
