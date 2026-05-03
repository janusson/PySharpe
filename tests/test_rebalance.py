"""Tests for the rebalance workflow."""

from __future__ import annotations

import json

import pytest
import pandas as pd

from pysharpe.execution.rebalance import build_rebalance_plan, format_rebalance_plan, main


def _write_artifacts(export_dir) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "demo_weights.txt").write_text(
        "ticker,weight\nAAPL,0.60\nMSFT,0.40\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "AAPL": [100.0, 110.0],
            "MSFT": [200.0, 210.0],
        }
    ).to_csv(export_dir / "demo_collated.csv", index=False)


def test_build_rebalance_plan_allocates_to_underweight_asset(tmp_path):
    export_dir = tmp_path / "exports"
    _write_artifacts(export_dir)

    plan = build_rebalance_plan(
        "demo",
        new_cash=100.0,
        holdings_mapping={"AAPL": 2},
        holdings_kind="shares",
        export_dir=export_dir,
    )

    buy_orders = plan.buy_orders

    assert list(buy_orders["ticker"]) == ["MSFT"]
    assert buy_orders.iloc[0]["recommended_allocation"] == pytest.approx(100.0)
    assert buy_orders.iloc[0]["recommended_shares"] == pytest.approx(100.0 / 210.0)
    assert "opportunity_score" in plan.scored_state.columns

    rendered = format_rebalance_plan(plan)
    assert "Rebalance plan for demo" in rendered
    assert "MSFT" in rendered
    assert "$100.00" in rendered


def test_rebalance_main_accepts_json_file(tmp_path, capsys):
    export_dir = tmp_path / "exports"
    _write_artifacts(export_dir)

    holdings_path = tmp_path / "holdings.json"
    holdings_path.write_text(json.dumps({"AAPL": 2}), encoding="utf-8")

    exit_code = main(
        [
            "--portfolio",
            "demo",
            "--holdings-json",
            str(holdings_path),
            "--holdings-kind",
            "shares",
            "--new-cash",
            "100",
            "--export-dir",
            str(export_dir),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Rebalance plan for demo" in output
    assert "MSFT" in output
