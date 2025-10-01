"""Portfolio definition helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set

from pysharpe.config import PySharpeSettings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioDefinition:
    name: str
    tickers: Sequence[str]
    path: Path

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError(f"Portfolio {self.name} has no tickers")

    @property
    def ticker_set(self) -> Set[str]:
        return set(self.tickers)


def read_tickers(path: Path) -> List[str]:
    tickers: list[str] = []
    seen: set[str] = set()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        cleaned = raw_line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        if cleaned not in seen:
            tickers.append(cleaned)
            seen.add(cleaned)

    return tickers


class PortfolioRepository:
    """Load portfolio CSV files from a directory."""

    def __init__(
        self,
        settings: PySharpeSettings | None = None,
        *,
        directory: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._directory = Path(directory or self.settings.portfolio_dir)
        self._portfolios: dict[str, PortfolioDefinition] = {}
        self.refresh()

    @property
    def directory(self) -> Path:
        return self._directory

    def refresh(self) -> None:
        self._portfolios.clear()
        directory = self.directory
        if not directory.exists():
            logger.warning("Portfolio directory not found: %s", directory)
            return

        for csv_path in sorted(directory.glob("*.csv")):
            try:
                tickers = read_tickers(csv_path)
            except FileNotFoundError:
                logger.error("Portfolio file missing: %s", csv_path)
                continue

            if not tickers:
                logger.warning("Skipping empty portfolio file: %s", csv_path.name)
                continue

            definition = PortfolioDefinition(
                name=csv_path.stem,
                tickers=tuple(tickers),
                path=csv_path,
            )
            self._portfolios[definition.name] = definition

    def list_portfolios(self) -> List[PortfolioDefinition]:
        return sorted(self._portfolios.values(), key=lambda item: item.name)

    def get_portfolio(self, name: str) -> PortfolioDefinition:
        if name in self._portfolios:
            return self._portfolios[name]

        # allow direct path lookups
        candidate = Path(name)
        if candidate.exists() and candidate.is_file():
            tickers = read_tickers(candidate)
            if not tickers:
                raise ValueError(f"Portfolio file {candidate} contains no tickers")
            return PortfolioDefinition(candidate.stem, tuple(tickers), candidate)

        raise FileNotFoundError(f"Portfolio file not found: {name}")

    def iter_definitions(self, names: Iterable[str] | None = None) -> Iterable[PortfolioDefinition]:
        if names is None:
            yield from self.list_portfolios()
            return

        for name in names:
            yield self.get_portfolio(name)
