"""Tests for PySharpe configuration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysharpe import config


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


def test_ensure_directories_create_all(tmp_path):
    settings = config.build_settings(base_dir=tmp_path)
    # Ensure directories do not exist beforehand
    for path in (
        settings.portfolio_dir,
        settings.price_history_dir,
        settings.export_dir,
        settings.info_dir,
        settings.log_dir,
    ):
        assert not path.exists()

    settings.ensure_directories()

    for path in (
        settings.portfolio_dir,
        settings.price_history_dir,
        settings.export_dir,
        settings.info_dir,
        settings.log_dir,
    ):
        assert path.exists()
