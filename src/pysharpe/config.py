"""Central configuration utilities for PySharpe."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_DEFAULT_DATA_DIR = Path("data")


@dataclass(frozen=True)
class PySharpeSettings:
    """Application-level settings with directory layout and behaviour toggles.

    Attributes:
        data_dir: Root directory for all generated artefacts.
        price_history_dir: Folder containing per-ticker CSV downloads.
        export_dir: Folder containing collated portfolios and optimisation outputs.
        portfolio_dir: Folder containing user-provided portfolio definitions.
        info_dir: Folder used by the info download helpers.
        log_dir: Folder for timestamped log files.
        log_level: Default logging threshold (INFO by default).
        artifact_version: Metadata version tag written to exports.

    Example:
        >>> from pysharpe.config import build_settings
        >>> settings = build_settings()
        >>> settings.export_dir.name
        'exports'
    """

    data_dir: Path = field(default_factory=lambda: _DEFAULT_DATA_DIR)
    price_history_dir: Path = field(init=False)
    export_dir: Path = field(init=False)
    portfolio_dir: Path = field(init=False)
    info_dir: Path = field(init=False)
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    log_level: str = "INFO"
    artifact_version: str = "v1"

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "data_dir", self.data_dir.resolve())
        object.__setattr__(self, "portfolio_dir", self.data_dir / "portfolio")
        object.__setattr__(self, "price_history_dir", self.data_dir / "price_hist")
        object.__setattr__(self, "export_dir", self.data_dir / "exports")
        object.__setattr__(self, "info_dir", self.data_dir / "info")
        log_dir = self.log_dir
        if not log_dir.is_absolute():
            log_dir = (self.data_dir / log_dir).resolve()
        else:
            log_dir = log_dir.resolve()
        object.__setattr__(self, "log_dir", log_dir)

    def ensure_directories(self) -> None:
        """Create the standard directory tree if it is missing.

        Example:
            >>> from pysharpe.config import build_settings
            >>> settings = build_settings()
            >>> settings.ensure_directories()  # Creates folders on disk
        """

        for directory in (
            self.portfolio_dir,
            self.price_history_dir,
            self.export_dir,
            self.info_dir,
            self.log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _path_from_env(var_name: str, default: Path) -> Path:
    override = os.getenv(var_name)
    if not override:
        return default
    return Path(override).expanduser()


def build_settings(base_dir: Path | None = None) -> PySharpeSettings:
    """Construct settings, honouring environment overrides when present.

    Args:
        base_dir: Optional override for the root data directory. When omitted the
            function respects the ``PYSHARPE_DATA_DIR`` environment variable or
            defaults to ``./data``.

    Returns:
        A fully initialised :class:`PySharpeSettings` instance.

    Example:
        >>> from pathlib import Path
        >>> from pysharpe.config import build_settings
        >>> custom = build_settings(base_dir=Path('tmp-data'))
        >>> custom.data_dir.name
        'tmp-data'
    """

    if base_dir is None:
        data_dir = _path_from_env("PYSHARPE_DATA_DIR", _DEFAULT_DATA_DIR)
        data_dir = data_dir.expanduser().resolve()
        log_dir_env = os.getenv("PYSHARPE_LOG_DIR")
        if log_dir_env:
            log_dir = Path(log_dir_env).expanduser().resolve()
        else:
            log_dir = data_dir / "logs"
    else:
        data_dir = Path(base_dir).expanduser().resolve()
        log_dir = data_dir / "logs"

    log_level = os.getenv("PYSHARPE_LOG_LEVEL", "INFO")
    artifact_version = os.getenv("PYSHARPE_ARTIFACT_VERSION", "v1")

    return PySharpeSettings(
        data_dir=data_dir,
        log_dir=log_dir,
        log_level=log_level,
        artifact_version=artifact_version,
    )


@lru_cache(maxsize=1)
def get_settings() -> PySharpeSettings:
    """Return the cached settings instance for the current process.

    Returns:
        The singleton :class:`PySharpeSettings` created via :func:`build_settings`.

    Example:
        >>> from pysharpe.config import get_settings
        >>> first = get_settings()
        >>> second = get_settings()
        >>> first is second
        True
    """

    return build_settings()


__all__ = ["PySharpeSettings", "build_settings", "get_settings"]
