"""Smoke-check the installed wheel from outside the source checkout.

Run with the Python interpreter from a fresh wheel-only virtual environment.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pysharpe


def main() -> None:
    package_path = Path(pysharpe.__file__).resolve()
    assert package_path.is_relative_to(Path(sys.prefix).resolve()), package_path

    for module in pkgutil.walk_packages(pysharpe.__path__, prefix="pysharpe."):
        if all(not part.startswith("_") for part in module.name.split(".")):
            importlib.import_module(module.name)

    for name in pysharpe._EXPORT_MAP:
        getattr(pysharpe, name)

    cli = Path(sys.prefix) / "bin" / "pysharpe"
    help_result = subprocess.run(
        [cli, "--help"], capture_output=True, text=True, check=True
    )
    assert "allocate" in help_result.stdout

    with TemporaryDirectory() as directory:
        portfolio = Path(directory) / "synthetic.csv"
        portfolio.write_text(
            "ticker,current_value,target_weight\nAAA,100,0.6\nBBB,100,0.4\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [cli, "allocate", "--portfolio", portfolio, "--amount", "1000"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Allocation recommendation for $1,000.00" in result.stdout
        assert "AAA" in result.stdout and "BBB" in result.stdout

    print("Wheel import, public exports, console script, and offline allocation: OK")


if __name__ == "__main__":
    main()
