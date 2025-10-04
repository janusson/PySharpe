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
    """Representation of a portfolio definition loaded from disk.

    Attributes:
        name: Portfolio identifier (usually the CSV stem).
        tickers: Ordered tickers included in the portfolio.
        path: Filesystem path to the source CSV file.

    Example:
        >>> from pathlib import Path
        >>> from pysharpe.data.portfolio import PortfolioDefinition
        >>> definition = PortfolioDefinition("growth", ("AAPL", "MSFT"), Path("growth.csv"))
        >>> definition.ticker_set
        {'AAPL', 'MSFT'}
    """

    name: str
    tickers: Sequence[str]
    path: Path

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError(f"Portfolio {self.name} has no tickers")

    @property
    def ticker_set(self) -> Set[str]:
        """Return the unique tickers in the portfolio.

        Example:
            >>> from pathlib import Path
            >>> from pysharpe.data.portfolio import PortfolioDefinition
            >>> PortfolioDefinition("demo", ("AAPL", "AAPL", "MSFT"), Path("demo.csv")).ticker_set
            {'AAPL', 'MSFT'}
        """

        return set(self.tickers)


def read_tickers(path: Path) -> List[str]:
    """Read tickers from a newline-delimited file.

    Args:
        path: File containing one ticker per line (``#`` lines are ignored).

    Returns:
        A list preserving the order of appearance while removing duplicates.

    Example:
        >>> from pathlib import Path
        >>> path = Path('example.csv')
        >>> _ = path.write_text('AAPL\n# comment\nMSFT\nAAPL\n', encoding='utf-8')
        >>> read_tickers(path)
        ['AAPL', 'MSFT']
        >>> path.unlink()
    """

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
    """Load portfolio CSV files from a directory.

    Example:
        >>> from pysharpe.data.portfolio import PortfolioRepository
        >>> repo = PortfolioRepository()
        >>> isinstance(repo.list_portfolios(), list)
        True
    """

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
        """Directory currently scanned for portfolio CSV files."""

        return self._directory

    def refresh(self) -> None:
        """Reload portfolio definitions from disk.

        Example:
            >>> from pysharpe.data.portfolio import PortfolioRepository
            >>> repo = PortfolioRepository()
            >>> repo.refresh()
        """

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
        """Return all discovered portfolio definitions sorted by name.

        Example:
            >>> from pysharpe.data.portfolio import PortfolioRepository
            >>> repo = PortfolioRepository()
            >>> isinstance(repo.list_portfolios(), list)
            True
        """

        return sorted(self._portfolios.values(), key=lambda item: item.name)

    def get_portfolio(self, name: str) -> PortfolioDefinition:
        """Retrieve a portfolio definition by name or direct path.

        Args:
            name: Portfolio identifier or filesystem path to a CSV.

        Returns:
            Matching :class:`PortfolioDefinition` instance.

        Raises:
            FileNotFoundError: If no matching CSV could be located.
            ValueError: If the resolved CSV is empty.

        Example:
            >>> from pathlib import Path
            >>> from pysharpe.data.portfolio import PortfolioRepository
            >>> repo = PortfolioRepository()
            >>> portfolio_dir = repo.directory
            >>> (portfolio_dir / 'demo.csv').write_text('AAPL', encoding='utf-8')
            >>> repo.refresh()
            >>> repo.get_portfolio('demo').name
            'demo'
        """

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
        """Yield portfolio definitions for the requested names.

        Args:
            names: Optional iterable of portfolio names. When omitted every
                definition discovered on disk is yielded.

        Yields:
            :class:`PortfolioDefinition` objects.

        Example:
            >>> from pysharpe.data.portfolio import PortfolioRepository
            >>> repo = PortfolioRepository()
            >>> list(repo.iter_definitions([]))
            []
        """

        if names is None:
            yield from self.list_portfolios()
            return

        for name in names:
            yield self.get_portfolio(name)
