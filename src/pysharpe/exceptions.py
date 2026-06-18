"""Domain-specific exception hierarchy for PySharpe.

These exceptions shield users from raw pandas, numpy, and yfinance stack
traces when inputs are missing, corrupted, or structurally flawed. Every
message is designed to be human-readable and actionable.
"""

from __future__ import annotations


class PySharpeError(Exception):
    """Base exception for all PySharpe domain errors.

    All custom exceptions in this package inherit from ``PySharpeError`` so
    that downstream code can catch a single type when guarding against
    expected failures.
    """


class DataIngestionError(PySharpeError):
    """Raised when data cannot be loaded, parsed, or is missing entirely.

    Typical causes:

    * A required optimisation artefact (weights file, collated CSV) is absent.
    * A CSV or JSON file is empty, truncated, or unparseable.
    * A file exists but uses an unexpected encoding or delimiter.
    """


class DataValidationError(PySharpeError):
    """Raised when loaded data fails structural or type-validation checks.

    Typical causes:

    * A required column (``ticker``, ``weight``, ``current_value``) is missing.
    * A numeric column contains text, non-finite, or negative values.
    * A ticker list is empty after filtering out blank or duplicate entries.
    """


class ExecutionConfigError(PySharpeError):
    """Raised when execution or portfolio configuration is invalid.

    Typical causes:

    * ``portfolio_config.json`` is missing, malformed, or contains unexpected types.
    * ``proxy_map.json`` has entries with missing required fields.
    * MER / geography mappings reference tickers not present in the portfolio.
    """
