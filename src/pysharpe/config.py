"""Central configuration utilities for PySharpe."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

_DEFAULT_DATA_DIR = Path("data")


@dataclass(frozen=True)
class PySharpeSettings:
    """Application-level settings with directory layout and behaviour toggles."""

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

    def ensure_directories(self) -> None:
        """Create core directories if needed."""

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


def build_settings(base_dir: Optional[Path] = None) -> PySharpeSettings:
    """Construct settings, honouring environment overrides where provided."""

    if base_dir is None:
        data_dir = _path_from_env("PYSHARPE_DATA_DIR", _DEFAULT_DATA_DIR)
    else:
        data_dir = base_dir

    log_dir = _path_from_env("PYSHARPE_LOG_DIR", Path("logs"))
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
    """Return a cached settings instance."""

    return build_settings()


__all__ = ["PySharpeSettings", "build_settings", "get_settings"]
