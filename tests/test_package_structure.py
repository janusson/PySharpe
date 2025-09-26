"""Smoke tests for the PySharpe package layout."""

import importlib

import pytest

pytest.importorskip("pandas")


@pytest.mark.parametrize(
    "module_name",
    [
        "pysharpe",
        "pysharpe.data.fetch",
        "pysharpe.models",
        "pysharpe.optimization.optimizer",
        "pysharpe.visualization.plotting",
    ],
)
def test_modules_importable(module_name: str) -> None:
    """Ensure the key package modules can be imported."""

    importlib.import_module(module_name)
