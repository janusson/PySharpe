"""Logging configuration helpers."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from pysharpe.config import get_settings


def configure_logging(log_dir: Path | None = None, level: str | None = None) -> Path:
    """Initialise file-based logging for PySharpe utilities.

    Args:
        log_dir: Optional destination folder for log files. Defaults to the
            value configured in :mod:`pysharpe.config`.
        level: Optional logging level string (for example ``"DEBUG"``).

    Returns:
        Path to the created log file.

    Example:
        >>> from pysharpe.logging_utils import configure_logging
        >>> log_file = configure_logging(level="INFO")
        >>> log_file.suffix
        '.log'
    """
    settings = get_settings()
    target_dir = log_dir or settings.log_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    log_path = target_dir / f"pysharpe_{timestamp}.log"

    logging.basicConfig(
        level=(level or settings.log_level).upper(),
        filename=str(log_path),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger(__name__).info("Logging initialised: %s", log_path)
    return log_path


__all__ = ["configure_logging"]
