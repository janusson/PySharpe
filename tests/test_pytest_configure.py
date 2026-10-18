"""Regression tests for the suite's collection-time Pyright hook."""

from __future__ import annotations

import conftest
import pytest


def test_pyright_failure_warns_without_using_removed_config_warn(monkeypatch):
    monkeypatch.setattr(conftest, "_pyright_available", lambda: True)
    monkeypatch.setattr(
        conftest,
        "_run_pyright",
        lambda: (-1, -1, "synthetic pyright runner failure"),
    )

    with pytest.warns(pytest.PytestWarning, match="synthetic pyright runner failure"):
        conftest.pytest_configure(pytest.Config.fromdictargs({}, []))
