"""Pin the synthetic rebalance example to its hand-derived buy plan.

The expected figures below are written out from the worked arithmetic in
``examples/rebalance_demo/README.md``; they are not recomputed with PySharpe
helpers. All inputs are fictional and no network access is involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pysharpe import cli
from pysharpe.config import load_execution_config
from pysharpe.execution.rebalance import build_rebalance_plan

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "rebalance_demo"

# ticker: (price, current weight, score, buy dollars, buy shares)
HAND_DERIVED = {
    "SYNA.TO": (50.0, 0.60, 0.2, 0.0, 0.0),
    "SYNB.TO": (25.0, 0.25, 0.5, 525.0, 21.0),
    "SYNC.TO": (40.0, 0.10, 0.8, 1040.0, 26.0),
    "SYND.TO": (100.0, 0.05, 0.5, 500.0, 5.0),
}
HAND_DERIVED_LEFTOVER = 85.0


def test_example_plan_matches_hand_derived_values() -> None:
    plan = build_rebalance_plan(
        "synthetic_demo",
        new_cash=2150.0,
        holdings_csv=EXAMPLE_DIR / "holdings.csv",
        export_dir=EXAMPLE_DIR / "exports",
        config_path=EXAMPLE_DIR / "config.json",
        execution_config=load_execution_config(EXAMPLE_DIR / "config.json"),
        proxy_map={},
    )

    rows = plan.allocations.set_index("ticker")
    assert set(rows.index) == set(HAND_DERIVED)
    for ticker, (price, weight, score, dollars, shares) in HAND_DERIVED.items():
        row = rows.loc[ticker]
        assert row["latest_price"] == pytest.approx(price)
        assert row["current_weight"] == pytest.approx(weight)
        assert row["opportunity_score"] == pytest.approx(score)
        assert row["recommended_allocation"] == pytest.approx(dollars)
        assert row["recommended_shares"] == pytest.approx(shares)

    assert rows["leftover_cash"].iloc[0] == pytest.approx(HAND_DERIVED_LEFTOVER)
    spent = rows["recommended_allocation"].sum()
    assert spent + HAND_DERIVED_LEFTOVER == pytest.approx(2150.0)


def test_example_cli_prints_hand_derived_plan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(
        [
            "rebalance",
            "--portfolio",
            "synthetic_demo",
            "--holdings-csv",
            str(EXAMPLE_DIR / "holdings.csv"),
            "--new-cash",
            "2150",
            "--export-dir",
            str(EXAMPLE_DIR / "exports"),
            "--config",
            str(EXAMPLE_DIR / "config.json"),
            "--include-zero-buys",
        ]
    )

    assert exit_code == 0
    lines = [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]
    assert "Unallocated (fractional leftover): $85.00" in lines
    assert "SYNC.TO $40.00 20.00% 10.00% 0.8000 $1,040.00 26.0000" in lines
    assert "SYNB.TO $25.00 30.00% 25.00% 0.5000 $525.00 21.0000" in lines
    assert "SYND.TO $100.00 10.00% 5.00% 0.5000 $500.00 5.0000" in lines
    assert "SYNA.TO $50.00 40.00% 60.00% 0.2000 $0.00 0.0000" in lines
