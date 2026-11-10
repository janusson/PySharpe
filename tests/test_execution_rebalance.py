"""Tests for the rebalance engine's input validation, allocation, and rendering.

.. note::

    **Canadian TFSA Constraint** — Plans are generated under the assumption of
    a Canadian Tax-Free Savings Account: capital gains are tax-exempt, so no
    tax-loss harvesting logic exists here.  Foreign withholding tax drag on
    US-domiciled assets inside a TFSA/FHSA is modelled as a score penalty.

    **Synthetic data only** — All prices, weights, and holdings are
    hand-constructed.  Tickers such as ``VFV.TO``/``VCN.TO`` are real
    CAD-denominated broad-market ETFs; ``VOO`` is used as a synthetic
    US-listed placeholder.  No network access occurs.

    **Money-moving paths** — ``build_rebalance_plan`` is the authoritative
    data-mutation path that turns saved optimisation artefacts into dollar
    buy orders.  These tests pin the exact arithmetic (FX fees, whole-share
    flooring, per-account cash splits) rather than merely the shape of the
    output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysharpe.config import ExecutionConfig
from pysharpe.exceptions import DataIngestionError, DataValidationError
from pysharpe.execution.rebalance import (
    RebalancePlan,
    _coerce_non_negative,
    _determine_account_tax_adjustment,
    _format_money,
    _format_percent,
    _format_shares,
    _load_allocation_inputs,
    _load_holdings_frame,
    _load_holdings_mapping,
    _load_latest_prices,
    _load_target_weights,
    _merge_fundamentals,
    _normalise_label,
    _prepare_ticker_frame,
    _rename_known_columns,
    build_rebalance_plan,
    format_rebalance_plan,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_artifacts(
    export_dir: Path,
    *,
    weights: dict[str, float],
    prices: dict[str, list[float]],
    portfolio: str = "demo",
) -> None:
    """Write the weights and collated price artefacts build_rebalance_plan reads."""

    export_dir.mkdir(parents=True, exist_ok=True)
    weight_rows = "\n".join(f"{ticker},{weight}" for ticker, weight in weights.items())
    (export_dir / f"{portfolio}_weights.txt").write_text(
        f"ticker,weight\n{weight_rows}\n",
        encoding="utf-8",
    )
    row_count = len(next(iter(prices.values())))
    frame = pd.DataFrame({"Date": [f"2024-01-0{i + 1}" for i in range(row_count)]})
    for ticker, series in prices.items():
        frame[ticker] = series
    frame.to_csv(export_dir / f"{portfolio}_collated.csv", index=False)


def _write_holdings(
    path: Path,
    rows: list[dict[str, object]],
    *,
    columns: list[str] | None = None,
) -> Path:
    """Write a holdings CSV from explicit row dicts."""

    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False)
    return path


def _latest_prices(prices: dict[str, float]) -> pd.DataFrame:
    """Build a latest-price lookup frame."""

    return pd.DataFrame(
        {
            "ticker": list(prices.keys()),
            "latest_price": list(prices.values()),
        }
    )


def _make_plan(
    allocations: pd.DataFrame,
    *,
    new_cash: float = 1000.0,
    account_allocations: dict[str, pd.DataFrame] | None = None,
    account_cash: dict[str, float] | None = None,
) -> RebalancePlan:
    """Construct a RebalancePlan for rendering tests only."""

    return RebalancePlan(
        portfolio_name="demo",
        new_cash=new_cash,
        weights_path=Path("demo_weights.txt"),
        prices_path=Path("demo_collated.csv"),
        scored_state=allocations.copy(),
        allocations=allocations,
        account_allocations=account_allocations,
        account_cash=account_cash,
    )


# ---------------------------------------------------------------------------
# Column normalisation and ticker preparation
# ---------------------------------------------------------------------------


def test_normalise_label_canonicalises_user_labels():
    assert _normalise_label(" Total-Value ") == "total_value"
    assert _normalise_label("Registration") == "registration"


def test_rename_known_columns_maps_every_alias_family():
    frame = pd.DataFrame(
        columns=[
            "Symbol",
            "Total Value",
            "Units",
            "Registration",
            "P/E Ratio",
            "pe ratio",
            "Shares",
        ]
    )

    renamed = _rename_known_columns(frame)

    assert set(renamed.columns) == {
        "ticker",
        "current_value",
        "shares",
        "account",
        "P/E Ratio",
        "pe_ratio",
    }


def test_coerce_non_negative_rejects_non_numeric_values():
    frame = pd.DataFrame(
        {"ticker": ["VFV.TO", "VCN.TO"], "current_value": [1.0, "oops"]}
    )

    with pytest.raises(DataValidationError, match="Non-numeric current value"):
        _coerce_non_negative(frame, "current_value", label="current value")


def test_coerce_non_negative_rejects_negative_values():
    frame = pd.DataFrame({"ticker": ["VFV.TO"], "current_value": [-5.0]})

    with pytest.raises(DataValidationError, match="cannot be negative"):
        _coerce_non_negative(frame, "current_value", label="current value")


def test_coerce_non_negative_returns_numeric_series():
    frame = pd.DataFrame({"ticker": ["VFV.TO"], "current_value": ["12.5"]})

    values = _coerce_non_negative(frame, "current_value", label="current value")

    assert values.tolist() == [12.5]


def test_prepare_ticker_frame_requires_ticker_column():
    with pytest.raises(DataValidationError, match="missing a ticker column"):
        _prepare_ticker_frame(pd.DataFrame({"current_value": [1.0]}))


def test_prepare_ticker_frame_rejects_blank_only_tickers():
    frame = pd.DataFrame({"ticker": ["", "  "], "current_value": [1.0, 2.0]})

    with pytest.raises(DataValidationError, match="no valid ticker entries"):
        _prepare_ticker_frame(frame)


def test_prepare_ticker_frame_strips_whitespace():
    frame = pd.DataFrame({"ticker": [" VFV.TO ", ""], "current_value": [1.0, 2.0]})

    prepared = _prepare_ticker_frame(frame)

    assert prepared["ticker"].tolist() == ["VFV.TO"]


# ---------------------------------------------------------------------------
# Artefact loaders
# ---------------------------------------------------------------------------


def _weights_dir(tmp_path: Path) -> Path:
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    return export_dir


def test_load_target_weights_missing_file(tmp_path):
    with pytest.raises(DataIngestionError, match="weights file not found"):
        _load_target_weights("nope", _weights_dir(tmp_path))


def test_load_target_weights_empty_file(tmp_path):
    export_dir = _weights_dir(tmp_path)
    (export_dir / "demo_weights.txt").write_text("", encoding="utf-8")

    with pytest.raises(DataIngestionError, match="is empty"):
        _load_target_weights("demo", export_dir)


def test_load_target_weights_requires_ticker_column(tmp_path):
    export_dir = _weights_dir(tmp_path)
    (export_dir / "demo_weights.txt").write_text("weight\n0.5\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="missing a ticker column"):
        _load_target_weights("demo", export_dir)


def test_load_target_weights_requires_weight_column(tmp_path):
    export_dir = _weights_dir(tmp_path)
    (export_dir / "demo_weights.txt").write_text(
        "ticker,allocation\nVFV.TO,0.5\n", encoding="utf-8"
    )

    with pytest.raises(DataValidationError, match="missing a weight column"):
        _load_target_weights("demo", export_dir)


def test_load_target_weights_renames_weight_and_fills_non_numeric(tmp_path):
    export_dir = _weights_dir(tmp_path)
    (export_dir / "demo_weights.txt").write_text(
        "ticker,weight\nVFV.TO,0.6\nVCN.TO,oops\n", encoding="utf-8"
    )

    frame, path = _load_target_weights("demo", export_dir)

    assert path == export_dir / "demo_weights.txt"
    assert frame["ticker"].tolist() == ["VFV.TO", "VCN.TO"]
    assert frame["target_weight"].tolist() == [0.6, 0.0]


def test_load_target_weights_parser_error(tmp_path, monkeypatch):
    export_dir = _weights_dir(tmp_path)
    (export_dir / "demo_weights.txt").write_text(
        "ticker,weight\nA,1,2\n", encoding="utf-8"
    )
    real_read_csv = pd.read_csv

    def _raise_parser_error(*args, **kwargs):
        raise pd.errors.ParserError("bad token")

    monkeypatch.setattr(pd, "read_csv", _raise_parser_error)

    with pytest.raises(DataIngestionError, match="Unable to parse weights file"):
        _load_target_weights("demo", export_dir)

    monkeypatch.setattr(pd, "read_csv", real_read_csv)


def test_load_target_weights_unicode_error(tmp_path, monkeypatch):
    export_dir = _weights_dir(tmp_path)
    (export_dir / "demo_weights.txt").write_text(
        "ticker,weight\nA,1\n", encoding="utf-8"
    )

    def _raise_unicode_error(*args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(pd, "read_csv", _raise_unicode_error)

    with pytest.raises(DataIngestionError, match="Cannot read weights file"):
        _load_target_weights("demo", export_dir)


def test_load_latest_prices_missing_file(tmp_path):
    with pytest.raises(DataIngestionError, match="Collated price history not found"):
        _load_latest_prices("nope", _weights_dir(tmp_path))


def test_load_latest_prices_empty_file(tmp_path):
    export_dir = _weights_dir(tmp_path)
    (export_dir / "demo_collated.csv").write_text("", encoding="utf-8")

    with pytest.raises(DataIngestionError, match="is empty"):
        _load_latest_prices("demo", export_dir)


def test_load_latest_prices_forward_fills_gaps(tmp_path):
    export_dir = _weights_dir(tmp_path)
    pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "VFV.TO": [100.0, np.nan],
            "VCN.TO": [50.0, 55.0],
            "Note": ["a", "b"],
        }
    ).to_csv(export_dir / "demo_collated.csv", index=False)

    latest, _ = _load_latest_prices("demo", export_dir)

    prices = dict(zip(latest["ticker"], latest["latest_price"], strict=True))
    assert prices == {"VFV.TO": 100.0, "VCN.TO": 55.0}


def test_load_latest_prices_all_nan_raises(tmp_path):
    export_dir = _weights_dir(tmp_path)
    pd.DataFrame(
        {"Date": ["2024-01-01"], "VFV.TO": [np.nan], "VCN.TO": [np.nan]}
    ).to_csv(export_dir / "demo_collated.csv", index=False)

    with pytest.raises(DataIngestionError, match="All rows may contain NaN"):
        _load_latest_prices("demo", export_dir)


def test_load_latest_prices_parser_error(tmp_path, monkeypatch):
    export_dir = _weights_dir(tmp_path)
    (export_dir / "demo_collated.csv").write_text(
        "Date,VFV.TO\n2024-01-01,100\n", encoding="utf-8"
    )

    def _raise_parser_error(*args, **kwargs):
        raise pd.errors.ParserError("bad token")

    monkeypatch.setattr(pd, "read_csv", _raise_parser_error)

    with pytest.raises(DataIngestionError, match="Unable to parse collated"):
        _load_latest_prices("demo", export_dir)


# ---------------------------------------------------------------------------
# Holdings mapping and holdings frame loaders
# ---------------------------------------------------------------------------


def test_load_holdings_mapping_parses_inline_json():
    assert _load_holdings_mapping('{"VFV.TO": 1000, " VCN.TO ": "250"}') == {
        "VFV.TO": 1000.0,
        "VCN.TO": 250.0,
    }


def test_load_holdings_mapping_reads_file(tmp_path):
    path = tmp_path / "holdings.json"
    path.write_text(json.dumps({"VFV.TO": 10}), encoding="utf-8")

    assert _load_holdings_mapping(str(path)) == {"VFV.TO": 10.0}


def test_load_holdings_mapping_unreadable_file(tmp_path):
    path = tmp_path / "holdings.json"
    path.mkdir()

    with pytest.raises(DataIngestionError, match="Cannot read holdings file"):
        _load_holdings_mapping(str(path))


def test_load_holdings_mapping_rejects_invalid_json():
    with pytest.raises(DataValidationError, match="not valid"):
        _load_holdings_mapping("{bad json")


def test_load_holdings_mapping_requires_object():
    with pytest.raises(DataValidationError, match="must be an object"):
        _load_holdings_mapping("[1, 2]")


def test_load_holdings_mapping_rejects_non_numeric_value():
    with pytest.raises(DataValidationError, match="must be a number"):
        _load_holdings_mapping('{"VFV.TO": "lots"}')


def test_load_holdings_mapping_rejects_negative_value():
    with pytest.raises(DataValidationError, match="cannot be negative"):
        _load_holdings_mapping('{"VFV.TO": -1}')


def test_load_holdings_mapping_skips_blank_tickers_and_rejects_empty():
    with pytest.raises(DataValidationError, match="any valid ticker entries"):
        _load_holdings_mapping('{" ": 100}')


def test_load_holdings_frame_requires_a_source():
    with pytest.raises(DataValidationError, match="No holdings provided"):
        _load_holdings_frame(
            holdings_csv=None,
            holdings_mapping=None,
            holdings_kind=None,
            latest_prices=_latest_prices({"VFV.TO": 100.0}),
        )


def test_load_holdings_frame_mapping_requires_kind():
    with pytest.raises(DataValidationError, match="holdings_kind is required"):
        _load_holdings_frame(
            holdings_csv=None,
            holdings_mapping={"VFV.TO": 1.0},
            holdings_kind=None,
            latest_prices=_latest_prices({"VFV.TO": 100.0}),
        )


def test_load_holdings_frame_mapping_shares_uses_latest_prices():
    frame = _load_holdings_frame(
        holdings_csv=None,
        holdings_mapping={"VFV.TO": 3.0},
        holdings_kind="shares",
        latest_prices=_latest_prices({"VFV.TO": 100.0}),
    )

    assert frame.loc[0, "current_shares"] == 3.0
    assert frame.loc[0, "current_value"] == 300.0


def test_load_holdings_frame_mapping_shares_requires_prices():
    with pytest.raises(DataValidationError, match="Missing latest prices"):
        _load_holdings_frame(
            holdings_csv=None,
            holdings_mapping={"VFV.TO": 3.0},
            holdings_kind="shares",
            latest_prices=_latest_prices({"VCN.TO": 50.0}),
        )


def test_load_holdings_frame_csv_requires_value_or_shares(tmp_path):
    path = _write_holdings(tmp_path / "h.csv", [{"ticker": "VFV.TO", "other": 1}])

    with pytest.raises(DataValidationError, match="current_value/total_value"):
        _load_holdings_frame(
            holdings_csv=path,
            holdings_mapping=None,
            holdings_kind=None,
            latest_prices=_latest_prices({"VFV.TO": 100.0}),
        )


def test_load_holdings_frame_renames_single_unknown_data_column(tmp_path):
    path = _write_holdings(tmp_path / "h.csv", [{"ticker": "VFV.TO", "units_held": 2}])

    frame = _load_holdings_frame(
        holdings_csv=path,
        holdings_mapping=None,
        holdings_kind="shares",
        latest_prices=_latest_prices({"VFV.TO": 100.0}),
    )

    assert frame.loc[0, "current_value"] == 200.0


def test_load_holdings_frame_rejects_ambiguous_extra_columns(tmp_path):
    path = _write_holdings(
        tmp_path / "h.csv",
        [{"ticker": "VFV.TO", "a": 1, "b": 2}],
    )

    with pytest.raises(DataValidationError, match="missing a current_value column"):
        _load_holdings_frame(
            holdings_csv=path,
            holdings_mapping=None,
            holdings_kind="value",
            latest_prices=_latest_prices({"VFV.TO": 100.0}),
        )

    with pytest.raises(DataValidationError, match="missing a shares column"):
        _load_holdings_frame(
            holdings_csv=path,
            holdings_mapping=None,
            holdings_kind="shares",
            latest_prices=_latest_prices({"VFV.TO": 100.0}),
        )


def test_load_holdings_frame_aggregates_duplicate_rows(tmp_path):
    path = _write_holdings(
        tmp_path / "h.csv",
        [
            {
                "ticker": "VFV.TO",
                "account": "TFSA",
                "market_value": 100.0,
                "pe_ratio": 18,
            },
            {
                "ticker": "VFV.TO",
                "account": "TFSA",
                "market_value": 50.0,
                "pe_ratio": 19,
            },
        ],
    )

    frame = _load_holdings_frame(
        holdings_csv=path,
        holdings_mapping=None,
        holdings_kind=None,
        latest_prices=_latest_prices({"VFV.TO": 100.0}),
    )

    assert len(frame) == 1
    assert frame.loc[0, "current_value"] == 150.0
    assert frame.loc[0, "pe_ratio"] == 18.0


def test_load_holdings_frame_rejects_negative_values(tmp_path):
    path = _write_holdings(
        tmp_path / "h.csv", [{"ticker": "VFV.TO", "current_value": -1.0}]
    )

    with pytest.raises(DataValidationError, match="cannot be negative"):
        _load_holdings_frame(
            holdings_csv=path,
            holdings_mapping=None,
            holdings_kind=None,
            latest_prices=_latest_prices({"VFV.TO": 100.0}),
        )


@pytest.mark.parametrize("account", ["", "   ", None])
def test_load_holdings_frame_rejects_blank_account_labels(tmp_path, account):
    """Regression: blank account labels previously reached ``new_cash / len([])``.

    A blank label is silently dropped from the account set, which left the
    multi-account equal-split branch dividing by zero during plan building.
    Blank labels must be rejected at load time with an actionable message.
    """

    path = _write_holdings(
        tmp_path / "h.csv",
        [
            {"ticker": "VFV.TO", "account": "TFSA", "current_value": 100.0},
            {"ticker": "VCN.TO", "account": account, "current_value": 50.0},
        ],
    )

    with pytest.raises(DataValidationError, match="missing an account label"):
        _load_holdings_frame(
            holdings_csv=path,
            holdings_mapping=None,
            holdings_kind=None,
            latest_prices=_latest_prices({"VFV.TO": 100.0, "VCN.TO": 50.0}),
        )


def test_load_holdings_frame_empty_csv(tmp_path):
    path = tmp_path / "h.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(DataIngestionError, match="Holdings CSV is empty"):
        _load_holdings_frame(
            holdings_csv=path,
            holdings_mapping=None,
            holdings_kind=None,
            latest_prices=_latest_prices({"VFV.TO": 100.0}),
        )


def test_load_holdings_frame_unreadable_csv(tmp_path):
    path = tmp_path / "holdings_dir"
    path.mkdir()

    with pytest.raises(DataIngestionError, match="Cannot read holdings CSV"):
        _load_holdings_frame(
            holdings_csv=path,
            holdings_mapping=None,
            holdings_kind=None,
            latest_prices=_latest_prices({"VFV.TO": 100.0}),
        )


# ---------------------------------------------------------------------------
# Allocation config loading and fundamentals merging
# ---------------------------------------------------------------------------


def test_load_allocation_inputs_none_returns_zero_weights():
    allocation_config, fundamentals = _load_allocation_inputs(None)

    assert allocation_config.weight_underweight == 0.0
    assert allocation_config.weight_valuation == 0.0
    assert fundamentals is None


def test_load_allocation_inputs_missing_file(tmp_path):
    with pytest.raises(DataIngestionError, match="not found"):
        _load_allocation_inputs(tmp_path / "missing.json")


def test_load_allocation_inputs_invalid_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(DataIngestionError, match="not valid JSON"):
        _load_allocation_inputs(path)


def test_load_allocation_inputs_filters_unknown_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "allocation_weights": {
                    "weight_underweight": 0.7,
                    "weight_valuation": 0.3,
                    "not_a_field": 99,
                },
                "fundamentals": {
                    "VFV.TO": {"pe_ratio": "18.5", "div_yield": 0.012},
                    "VCN.TO": {"unknown_metric": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    allocation_config, fundamentals = _load_allocation_inputs(path)

    assert allocation_config.weight_underweight == 0.7
    assert allocation_config.weight_valuation == 0.3
    assert fundamentals is not None
    assert fundamentals["ticker"].tolist() == ["VFV.TO", "VCN.TO"]
    assert fundamentals["pe_ratio"].iloc[0] == 18.5
    assert fundamentals["div_yield"].iloc[1] != fundamentals["div_yield"].iloc[1]


def test_load_allocation_inputs_ignores_empty_fundamentals(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"fundamentals": {}}), encoding="utf-8")

    _, fundamentals = _load_allocation_inputs(path)

    assert fundamentals is None


def test_merge_fundamentals_fills_and_adds_valuation_columns():
    frame = pd.DataFrame(
        {
            "ticker": ["VFV.TO", "VCN.TO"],
            "div_yield": [np.nan, 0.03],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "ticker": ["VFV.TO"],
            "div_yield": [0.012],
            "pe_ratio": [18.0],
        }
    )

    merged = _merge_fundamentals(frame, fundamentals)

    assert "div_yield_cfg" not in merged.columns
    assert merged.loc[0, "div_yield"] == 0.012
    assert merged.loc[1, "div_yield"] == 0.03
    assert merged.loc[0, "pe_ratio"] == 18.0


def test_merge_fundamentals_without_fundamentals_is_identity():
    frame = pd.DataFrame({"ticker": ["VFV.TO"]})

    assert _merge_fundamentals(frame, None) is frame


# ---------------------------------------------------------------------------
# Legacy tax-location adjustment
# ---------------------------------------------------------------------------


def test_tax_adjustment_for_us_domiciled_asset_in_tfsa():
    proxy_map = {"VOO": {"is_us_domiciled": True, "is_cad_denominated": False}}

    assert _determine_account_tax_adjustment("tfsa", "VOO", proxy_map) == pytest.approx(
        0.97
    )
    assert _determine_account_tax_adjustment("FHSA", "VOO", proxy_map) == pytest.approx(
        0.97
    )


@pytest.mark.parametrize("account", ["RRSP", "Non-Reg", "TFSA"])
def test_tax_adjustment_is_neutral_for_cad_assets(account):
    assert _determine_account_tax_adjustment(account, "VFV.TO") == 1.0


def test_tax_adjustment_is_neutral_for_us_asset_in_registered_account():
    proxy_map = {"VOO": {"is_us_domiciled": True, "is_cad_denominated": False}}

    assert _determine_account_tax_adjustment("RRSP", "VOO", proxy_map) == 1.0


def test_tax_adjustment_without_proxy_map_uses_fallback_metadata():
    # VOO has no .TO suffix, so the fallback marks it non-CAD but not US-domiciled.
    assert _determine_account_tax_adjustment("TFSA", "VOO") == 1.0


# ---------------------------------------------------------------------------
# Single-account plan building
# ---------------------------------------------------------------------------


def test_build_rebalance_plan_requires_positive_new_cash(tmp_path):
    _write_artifacts(
        tmp_path,
        weights={"VFV.TO": 1.0},
        prices={"VFV.TO": [100.0]},
    )

    with pytest.raises(DataValidationError, match="new_cash must be a positive"):
        build_rebalance_plan(
            "demo",
            new_cash=0,
            holdings_mapping={"VFV.TO": 0.0},
            holdings_kind="value",
            export_dir=tmp_path,
        )


def test_build_rebalance_plan_applies_fx_fee_to_non_cad_purchase(tmp_path):
    _write_artifacts(
        tmp_path,
        weights={"VFV.TO": 0.5, "VOO": 0.5},
        prices={"VFV.TO": [100.0], "VOO": [50.0]},
    )

    plan = build_rebalance_plan(
        "demo",
        new_cash=1000.0,
        holdings_mapping={"VFV.TO": 1000.0, "VOO": 0.0},
        holdings_kind="value",
        export_dir=tmp_path,
        execution_config=ExecutionConfig(fx_fee_bps=200),
    )

    row = plan.allocations.set_index("ticker").loc["VOO"]
    assert row["recommended_allocation"] == pytest.approx(980.0)
    assert row["recommended_shares"] == pytest.approx(19.6)


def test_build_rebalance_plan_leaves_cad_purchase_unaffected_by_fx_fee(tmp_path):
    _write_artifacts(
        tmp_path,
        weights={"VFV.TO": 0.5, "VOO": 0.5},
        prices={"VFV.TO": [100.0], "VOO": [50.0]},
    )

    plan = build_rebalance_plan(
        "demo",
        new_cash=1000.0,
        holdings_mapping={"VFV.TO": 0.0, "VOO": 1000.0},
        holdings_kind="value",
        export_dir=tmp_path,
        execution_config=ExecutionConfig(fx_fee_bps=200),
    )

    row = plan.allocations.set_index("ticker").loc["VFV.TO"]
    assert row["recommended_allocation"] == pytest.approx(1000.0)


def test_build_rebalance_plan_floors_whole_shares_and_tracks_leftover(tmp_path):
    _write_artifacts(
        tmp_path,
        weights={"VOO": 1.0},
        prices={"VOO": [30.0]},
    )

    plan = build_rebalance_plan(
        "demo",
        new_cash=1000.0,
        holdings_mapping={"VOO": 0.0},
        holdings_kind="value",
        export_dir=tmp_path,
        execution_config=ExecutionConfig(allow_fractional=False),
    )

    row = plan.allocations.set_index("ticker").loc["VOO"]
    assert row["recommended_shares"] == 33.0
    assert row["recommended_allocation"] == pytest.approx(990.0)
    assert row["leftover_cash"] == pytest.approx(10.0)


def test_build_rebalance_plan_fractional_shares_leave_no_leftover_column(tmp_path):
    _write_artifacts(
        tmp_path,
        weights={"VOO": 1.0},
        prices={"VOO": [30.0]},
    )

    plan = build_rebalance_plan(
        "demo",
        new_cash=1000.0,
        holdings_mapping={"VOO": 0.0},
        holdings_kind="value",
        export_dir=tmp_path,
    )

    assert "leftover_cash" not in plan.allocations.columns
    row = plan.allocations.set_index("ticker").loc["VOO"]
    assert row["recommended_shares"] == pytest.approx(1000.0 / 30.0)


def test_build_rebalance_plan_reports_post_buy_weight(tmp_path):
    _write_artifacts(
        tmp_path,
        weights={"VFV.TO": 1.0},
        prices={"VFV.TO": [10.0]},
    )

    plan = build_rebalance_plan(
        "demo",
        new_cash=1000.0,
        holdings_mapping={"VFV.TO": 1000.0},
        holdings_kind="value",
        export_dir=tmp_path,
    )

    row = plan.allocations.set_index("ticker").loc["VFV.TO"]
    assert row["estimated_post_buy_weight"] == pytest.approx(1.0)


def test_build_rebalance_plan_accepts_share_based_holdings(tmp_path):
    _write_artifacts(
        tmp_path,
        weights={"VFV.TO": 1.0},
        prices={"VFV.TO": [100.0]},
    )

    plan = build_rebalance_plan(
        "demo",
        new_cash=100.0,
        holdings_mapping={"VFV.TO": 2.0},
        holdings_kind="shares",
        export_dir=tmp_path,
    )

    assert plan.scored_state.set_index("ticker").loc["VFV.TO", "current_value"] == 200.0


def test_build_rebalance_plan_rejects_blank_accounts_end_to_end(tmp_path):
    """Regression: blank account labels must not reach the cash-split branch."""

    _write_artifacts(
        tmp_path,
        weights={"VFV.TO": 1.0},
        prices={"VFV.TO": [100.0]},
    )
    holdings = _write_holdings(
        tmp_path / "holdings.csv",
        [
            {"ticker": "VFV.TO", "account": "TFSA", "current_value": 100.0},
            {"ticker": "VFV.TO", "account": "", "current_value": 100.0},
        ],
    )

    with pytest.raises(DataValidationError, match="missing an account label"):
        build_rebalance_plan(
            "demo",
            new_cash=100.0,
            holdings_csv=holdings,
            export_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# Multi-account plan building
# ---------------------------------------------------------------------------


def test_build_rebalance_plan_splits_cash_pro_rata(tmp_path):
    _write_artifacts(
        tmp_path,
        weights={"VFV.TO": 1.0},
        prices={"VFV.TO": [100.0]},
    )
    holdings = _write_holdings(
        tmp_path / "holdings.csv",
        [
            {"ticker": "VFV.TO", "account": "TFSA", "current_value": 3000.0},
            {"ticker": "VFV.TO", "account": "RRSP", "current_value": 1000.0},
        ],
    )

    plan = build_rebalance_plan(
        "demo",
        new_cash=1000.0,
        holdings_csv=holdings,
        export_dir=tmp_path,
    )

    assert plan.is_multi_account
    assert plan.account_cash == {"RRSP": 250.0, "TFSA": 750.0}
    assert plan.accounts == ["RRSP", "TFSA"]
    assert plan.account_allocations["RRSP"]["recommended_allocation"].sum() == (
        pytest.approx(250.0)
    )
    assert plan.account_allocations["TFSA"]["recommended_allocation"].sum() == (
        pytest.approx(750.0)
    )


def test_build_rebalance_plan_equal_splits_when_accounts_are_empty(tmp_path):
    _write_artifacts(
        tmp_path,
        weights={"VFV.TO": 1.0},
        prices={"VFV.TO": [100.0]},
    )
    holdings = _write_holdings(
        tmp_path / "holdings.csv",
        [
            {"ticker": "VFV.TO", "account": "TFSA", "current_value": 0.0},
            {"ticker": "VFV.TO", "account": "RRSP", "current_value": 0.0},
        ],
    )

    plan = build_rebalance_plan(
        "demo",
        new_cash=1000.0,
        holdings_csv=holdings,
        export_dir=tmp_path,
    )

    assert plan.account_cash == {"RRSP": 500.0, "TFSA": 500.0}


def test_build_rebalance_plan_assigns_rounding_remainder_to_largest(tmp_path):
    _write_artifacts(
        tmp_path,
        weights={"VFV.TO": 1.0},
        prices={"VFV.TO": [100.0]},
    )
    holdings = _write_holdings(
        tmp_path / "holdings.csv",
        [
            {"ticker": "VFV.TO", "account": account, "current_value": 1000.0}
            for account in ("TFSA", "RRSP", "FHSA")
        ],
    )

    plan = build_rebalance_plan(
        "demo",
        new_cash=100.0,
        holdings_csv=holdings,
        export_dir=tmp_path,
    )

    assert plan.account_cash is not None
    assert sum(plan.account_cash.values()) == pytest.approx(100.0)
    # Accounts are processed in sorted order, so the first (FHSA) absorbs $0.01.
    assert plan.account_cash["FHSA"] == pytest.approx(33.34)
    assert plan.account_cash["RRSP"] == pytest.approx(33.33)
    assert plan.account_cash["TFSA"] == pytest.approx(33.33)


def test_build_rebalance_plan_penalises_us_assets_in_tfsa_scores(tmp_path):
    _write_artifacts(
        tmp_path,
        weights={"VOO": 1.0},
        prices={"VOO": [100.0]},
    )
    holdings = _write_holdings(
        tmp_path / "holdings.csv",
        [
            {"ticker": "VOO", "account": "TFSA", "current_value": 1000.0},
            {"ticker": "VOO", "account": "RRSP", "current_value": 1000.0},
        ],
    )

    plan = build_rebalance_plan(
        "demo",
        new_cash=100.0,
        holdings_csv=holdings,
        export_dir=tmp_path,
        proxy_map={"VOO": {"is_us_domiciled": True, "is_cad_denominated": False}},
    )

    scores = plan.scored_state.set_index(["account", "ticker"])["opportunity_score"]
    assert scores[("TFSA", "VOO")] == pytest.approx(0.97 * scores[("RRSP", "VOO")])


def test_build_rebalance_plan_applies_fx_fee_and_whole_shares_per_account(tmp_path):
    _write_artifacts(
        tmp_path,
        weights={"VOO": 1.0},
        prices={"VOO": [30.0]},
    )
    holdings = _write_holdings(
        tmp_path / "holdings.csv",
        [
            {"ticker": "VOO", "account": "TFSA", "current_value": 0.0},
            {"ticker": "VOO", "account": "RRSP", "current_value": 0.0},
        ],
    )

    plan = build_rebalance_plan(
        "demo",
        new_cash=1000.0,
        holdings_csv=holdings,
        export_dir=tmp_path,
        execution_config=ExecutionConfig(fx_fee_bps=200, allow_fractional=False),
    )

    assert plan.account_allocations is not None
    for account in ("RRSP", "TFSA"):
        allocation = plan.account_allocations[account].iloc[0]
        # $500 gross -> $490 after the 2% FX fee -> 16 shares at $30 = $480.
        assert allocation["recommended_allocation"] == pytest.approx(480.0)
        assert allocation["recommended_shares"] == 16.0
        assert allocation["leftover_cash"] == pytest.approx(10.0)
    assert plan.allocations["recommended_allocation"].sum() == pytest.approx(960.0)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _single_allocation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["VFV.TO"],
            "latest_price": [100.0],
            "target_weight": [1.0],
            "current_weight": [0.0],
            "opportunity_score": [0.8],
            "recommended_allocation": [1000.0],
            "recommended_shares": [10.0],
        }
    )


def test_formatters_handle_none_and_nan():
    assert _format_money(None) == "n/a"
    assert _format_money(float("nan")) == "n/a"
    assert _format_money(1234.5) == "$1,234.50"
    assert _format_percent(None) == "n/a"
    assert _format_percent(0.125) == "12.50%"
    assert _format_shares(float("nan")) == "n/a"
    assert _format_shares(10.0) == "10.0000"


def test_format_rebalance_plan_renders_single_account_buys():
    plan = _make_plan(_single_allocation_frame())

    text = format_rebalance_plan(plan)

    assert "Rebalance plan for demo" in text
    assert "New cash: $1,000.00" in text
    assert "VFV.TO" in text
    assert "$1,000.00" in text


def test_format_rebalance_plan_reports_no_buys():
    frame = _single_allocation_frame()
    frame["recommended_allocation"] = 0.0

    text = format_rebalance_plan(_make_plan(frame))

    assert "No buys recommended." in text


def test_format_rebalance_plan_include_zero_buys_shows_rows():
    frame = _single_allocation_frame()
    frame["recommended_allocation"] = 0.0

    text = format_rebalance_plan(_make_plan(frame), include_zero_buys=True)

    assert "VFV.TO" in text
    assert "No buys recommended." not in text


def test_format_rebalance_plan_reports_fractional_leftover():
    frame = _single_allocation_frame()
    frame["leftover_cash"] = 10.0

    text = format_rebalance_plan(_make_plan(frame))

    assert "Unallocated (fractional leftover): $10.00" in text


def test_format_rebalance_plan_groups_multi_account_orders():
    tfsa = _single_allocation_frame()
    rrsp = _single_allocation_frame()
    plan = _make_plan(
        _single_allocation_frame(),
        account_allocations={"RRSP": rrsp, "TFSA": tfsa},
        account_cash={"RRSP": 250.0, "TFSA": 750.0},
    )

    text = format_rebalance_plan(plan)

    assert "\u2500\u2500 RRSP ($250.00) \u2500\u2500" in text
    assert "\u2500\u2500 TFSA ($750.00) \u2500\u2500" in text


def test_format_rebalance_plan_reports_no_buys_per_account():
    frame = _single_allocation_frame()
    frame["recommended_allocation"] = 0.0
    plan = _make_plan(
        frame,
        account_allocations={"TFSA": frame.copy()},
        account_cash={"TFSA": 0.0},
    )

    text = format_rebalance_plan(plan)

    assert "  No buys recommended for this account." in text


def test_rebalance_plan_buy_orders_filters_zero_allocations():
    frame = _single_allocation_frame()
    frame.loc[1] = frame.loc[0]
    frame.loc[1, "ticker"] = "VCN.TO"
    frame.loc[1, "recommended_allocation"] = 0.0

    plan = _make_plan(frame)

    assert plan.buy_orders["ticker"].tolist() == ["VFV.TO"]
    assert not plan.is_multi_account
    assert plan.accounts == []


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------


def test_main_returns_error_when_artifacts_are_missing(tmp_path, capsys):
    """Regression: PySharpeError subclasses are not ValueError subclasses.

    Missing artefacts raise ``DataIngestionError``; before the fix the
    standalone ``rebalance`` entry point let it escape as a raw traceback.
    """

    exit_code = main(
        [
            "--portfolio",
            "missing",
            "--holdings-json",
            '{"VFV.TO": 100}',
            "--holdings-kind",
            "value",
            "--new-cash",
            "100",
            "--export-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out.startswith("Error:")
    assert "weights file not found" in captured.out


def test_main_returns_error_for_domain_validation_failure(tmp_path, capsys):
    """Regression: ``DataValidationError`` must map to exit code 1, not a traceback."""

    _write_artifacts(
        tmp_path,
        weights={"VFV.TO": 1.0},
        prices={"VFV.TO": [100.0]},
    )
    holdings = _write_holdings(
        tmp_path / "holdings.csv",
        [
            {"ticker": "VFV.TO", "account": "TFSA", "current_value": 100.0},
            {"ticker": "VFV.TO", "account": "  ", "current_value": 100.0},
        ],
    )

    exit_code = main(
        [
            "--portfolio",
            "demo",
            "--holdings-csv",
            str(holdings),
            "--new-cash",
            "100",
            "--export-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out.startswith("Error:")
    assert "missing an account label" in captured.out


def test_main_returns_error_for_invalid_holdings_json(capsys):
    exit_code = main(
        [
            "--portfolio",
            "demo",
            "--holdings-json",
            "{bad json",
            "--holdings-kind",
            "value",
            "--new-cash",
            "100",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Holdings JSON is not valid" in captured.out


def test_main_prints_plan_on_success(tmp_path, capsys):
    _write_artifacts(
        tmp_path,
        weights={"VFV.TO": 1.0},
        prices={"VFV.TO": [100.0]},
    )

    exit_code = main(
        [
            "--portfolio",
            "demo",
            "--holdings-json",
            '{"VFV.TO": 0}',
            "--holdings-kind",
            "value",
            "--new-cash",
            "100",
            "--export-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Rebalance plan for demo" in captured.out
