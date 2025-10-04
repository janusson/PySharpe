"""Tests for logging configuration helpers."""

from __future__ import annotations

import logging

from pysharpe import logging_utils


def test_configure_logging_uses_target_directory(tmp_path, monkeypatch):
    records: dict[str, object] = {}

    def fake_basicConfig(**kwargs):
        records.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basicConfig)

    # Use the real logging logger to avoid interfering with pytest's logging hooks.
    root = logging.getLogger()
    existing_handlers = list(root.handlers)
    for handler in existing_handlers:
        root.removeHandler(handler)

    log_path = logging_utils.configure_logging(log_dir=tmp_path, level="debug")

    assert log_path.parent == tmp_path
    assert log_path.suffix == ".log"
    assert records.get("level") == "DEBUG"
    assert "filename" in records

    # Restore original handlers to avoid leaking state across tests.
    for handler in root.handlers:
        root.removeHandler(handler)
    for handler in existing_handlers:
        root.addHandler(handler)
