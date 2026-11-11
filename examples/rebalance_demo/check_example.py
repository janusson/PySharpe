"""Check the synthetic rebalance example against independently derived values.

SYNTHETIC DATA ONLY. The tickers, prices, holdings, and target weights in this
directory are invented for demonstration. Nothing here is market data or
investment advice, and nothing is downloaded.

This script deliberately does not import ``pysharpe``. It:

1. recomputes the expected buy plan from the input files with exact rational
   arithmetic (standard library only),
2. runs the installed ``pysharpe rebalance`` console script as a subprocess,
3. prints expected versus observed values and exits non-zero on any mismatch.

Usage (from any working directory):

    python path/to/examples/rebalance_demo/check_example.py
    python path/to/examples/rebalance_demo/check_example.py --pysharpe /path/to/pysharpe
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from math import floor
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPORT_DIR = HERE / "exports"
PORTFOLIO = "synthetic_demo"
NEW_CASH = Fraction(2150)

# Documented scoring blend: 60% drift + 40% valuation. The example supplies no
# valuation inputs, so every asset gets the neutral valuation score of 0.5.
DRIFT_WEIGHT = Fraction(6, 10)
VALUATION_WEIGHT = Fraction(4, 10)
NEUTRAL_VALUATION = Fraction(1, 2)
MIN_ORDER_DOLLARS = Fraction(25)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expected_plan() -> dict[str, object]:
    """Derive the buy plan by hand-checkable arithmetic from the input files."""

    targets = {
        row["ticker"]: Fraction(row["weight"])
        for row in _read_rows(EXPORT_DIR / f"{PORTFOLIO}_weights.txt")
    }
    last_prices_row = _read_rows(EXPORT_DIR / f"{PORTFOLIO}_collated.csv")[-1]
    prices = {ticker: Fraction(last_prices_row[ticker]) for ticker in targets}
    shares = {
        row["ticker"]: Fraction(row["shares"])
        for row in _read_rows(HERE / "holdings.csv")
    }
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    assert config["allow_fractional"] is False, "example assumes whole shares"

    values = {ticker: shares[ticker] * prices[ticker] for ticker in targets}
    total_value = sum(values.values())
    current = {ticker: values[ticker] / total_value for ticker in targets}

    # Buy-only: an asset's gap is how far it sits below target (never negative).
    gaps = {
        ticker: max(targets[ticker] - current[ticker], Fraction(0))
        for ticker in targets
    }
    total_gap = sum(gaps.values())
    largest_gap = max(gaps.values())

    rows: dict[str, dict[str, Fraction]] = {}
    spent = Fraction(0)
    for ticker in targets:
        dollars = NEW_CASH * gaps[ticker] / total_gap
        # No order in this example falls under the $25 minimum-order threshold.
        assert dollars == 0 or dollars >= MIN_ORDER_DOLLARS
        buy_shares = Fraction(floor(dollars / prices[ticker]))
        buy_dollars = buy_shares * prices[ticker]
        spent += buy_dollars
        rows[ticker] = {
            "price": prices[ticker],
            "target_pct": targets[ticker] * 100,
            "current_pct": current[ticker] * 100,
            "score": DRIFT_WEIGHT * gaps[ticker] / largest_gap
            + VALUATION_WEIGHT * NEUTRAL_VALUATION,
            "buy_dollars": buy_dollars,
            "buy_shares": buy_shares,
        }
    return {"rows": rows, "leftover": NEW_CASH - spent}


def _number(text: str) -> Fraction:
    return Fraction(text.replace("$", "").replace(",", "").replace("%", ""))


def observed_plan(pysharpe: str) -> tuple[dict[str, object], list[str]]:
    """Run the installed CLI and parse its printed table."""

    command = [
        pysharpe,
        "rebalance",
        "--portfolio",
        PORTFOLIO,
        "--holdings-csv",
        str(HERE / "holdings.csv"),
        "--new-cash",
        str(NEW_CASH),
        "--export-dir",
        str(EXPORT_DIR),
        "--config",
        str(HERE / "config.json"),
        "--include-zero-buys",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.exit(
            f"pysharpe exited with {result.returncode}:\n{result.stdout}{result.stderr}"
        )

    rows: dict[str, dict[str, Fraction]] = {}
    leftover = Fraction(0)
    for line in result.stdout.splitlines():
        match = re.match(r"Unallocated \(fractional leftover\): (\S+)", line)
        if match:
            leftover = _number(match.group(1))
        parts = line.split()
        if len(parts) == 7 and parts[0].startswith("SYN"):
            keys = (
                "price",
                "target_pct",
                "current_pct",
                "score",
                "buy_dollars",
                "buy_shares",
            )
            rows[parts[0]] = dict(zip(keys, map(_number, parts[1:]), strict=True))
    return {"rows": rows, "leftover": leftover}, command


def _find_pysharpe(explicit: str | None) -> str:
    if explicit:
        return explicit
    beside_python = Path(sys.executable).parent / "pysharpe"
    if beside_python.exists():
        return str(beside_python)
    on_path = shutil.which("pysharpe")
    if on_path:
        return on_path
    sys.exit("Could not find the 'pysharpe' console script; pass --pysharpe PATH.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pysharpe", help="Path to the pysharpe console script.")
    args = parser.parse_args()

    pysharpe = _find_pysharpe(args.pysharpe)
    expected = expected_plan()
    observed, command = observed_plan(pysharpe)

    print("SYNTHETIC DATA - fictional tickers and prices, no network access")
    print(f"CLI under test: {pysharpe}")
    print(f"Command: {' '.join(command)}\n")
    print(f"{'ticker':<9}{'field':<13}{'expected':>12}{'observed':>12}  result")

    failures = 0

    def compare(ticker: str, field: str, want: Fraction, got: Fraction | None) -> None:
        nonlocal failures
        # The CLI prints 2 decimals for money/percent and 4 for score/shares.
        ok = got is not None and abs(want - got) <= Fraction(1, 200)
        failures += not ok
        shown = "missing" if got is None else f"{float(got):,.4f}"
        status = "ok" if ok else "MISMATCH"
        print(f"{ticker:<9}{field:<13}{float(want):>12,.4f}{shown:>12}  {status}")

    expected_rows: dict[str, dict[str, Fraction]] = expected["rows"]  # type: ignore[assignment]
    observed_rows: dict[str, dict[str, Fraction]] = observed["rows"]  # type: ignore[assignment]
    for ticker, fields in expected_rows.items():
        for field, want in fields.items():
            compare(ticker, field, want, observed_rows.get(ticker, {}).get(field))
    compare("(plan)", "leftover", expected["leftover"], observed["leftover"])  # type: ignore[arg-type]

    total = sum(len(fields) for fields in expected_rows.values()) + 1
    print(f"\n{total - failures}/{total} values match.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
