"""Tests for PySharpe configuration helpers.

.. note::

    **Canadian Investment Compliance** — Default MER values must be decimal
    fractions (< 0.10), never percentage points.  A real Canadian ETF MER
    expressed as a decimal fraction is always well below 0.10 (e.g., 0.0017
    for 0.17%).  Values like 0.17 signal the old percentage-point convention
    and are rejected.

    ``get_settings()`` is LRU-cached; call ``cache_clear()`` in tests
    that vary environment variables.  ``PYSHARPE_DATA_DIR`` overrides the
    data root.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pysharpe import config, logging_utils


@pytest.fixture(autouse=True)
def clear_settings_cache():
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_build_settings_from_environment(monkeypatch, tmp_path):
    data_dir = tmp_path / "custom_data"
    log_dir = tmp_path / "custom_logs"
    monkeypatch.setenv("PYSHARPE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("PYSHARPE_LOG_DIR", str(log_dir))
    monkeypatch.setenv("PYSHARPE_LOG_LEVEL", "debug")
    monkeypatch.setenv("PYSHARPE_ARTIFACT_VERSION", "test")

    settings = config.build_settings()

    assert settings.data_dir == data_dir.resolve()
    assert settings.cache_dir == data_dir.resolve() / "cache"
    assert settings.log_dir == log_dir.expanduser()
    assert settings.log_level == "debug"
    assert settings.artifact_version == "test"


def test_get_settings_returns_cached_instance(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSHARPE_DATA_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    first = config.get_settings()
    second = config.get_settings()
    assert first is second
    assert isinstance(first.data_dir, Path)


def test_default_mer_by_ticker_values_are_decimal_fractions():
    """Default MER values must be decimal fractions, not percentage points.

    A real ETF MER expressed as a decimal fraction is always well below 0.10
    (i.e. below 10%). A value of 0.17 would indicate 17% MER — impossible for
    a listed ETF — which signals the old percentage-point convention.
    """
    settings = config.build_settings()
    for ticker, mer in settings.mer_by_ticker.items():
        assert mer < 0.10, (
            f"MER for {ticker} is {mer}, which looks like a percentage point "
            f"({mer * 100:.2f}%). Use decimal fractions (e.g. 0.0017 for 0.17%)."
        )


def test_ensure_directories_create_all(tmp_path):
    settings = config.build_settings(base_dir=tmp_path)
    # Ensure directories do not exist beforehand
    for path in (
        settings.cache_dir,
        settings.portfolio_dir,
        settings.price_history_dir,
        settings.export_dir,
        settings.info_dir,
        settings.log_dir,
    ):
        assert not path.exists()

    settings.ensure_directories()

    for path in (
        settings.cache_dir,
        settings.portfolio_dir,
        settings.price_history_dir,
        settings.export_dir,
        settings.info_dir,
        settings.log_dir,
    ):
        assert path.exists()


def test_configure_logging_uses_target_directory(tmp_path, monkeypatch):
    records: dict[str, object] = {}

    def fake_basicConfig(**kwargs):
        records.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basicConfig)

    root = logging.getLogger()
    existing_handlers = list(root.handlers)
    for handler in existing_handlers:
        root.removeHandler(handler)

    log_path = logging_utils.configure_logging(log_dir=tmp_path, level="debug")

    assert log_path.parent == tmp_path
    assert log_path.suffix == ".log"
    assert records.get("level") == "DEBUG"
    assert "filename" in records

    for handler in root.handlers:
        root.removeHandler(handler)
    for handler in existing_handlers:
        root.addHandler(handler)
