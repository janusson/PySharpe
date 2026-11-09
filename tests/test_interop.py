"""Tests for the vendor-neutral holdings snapshot import.

Synthetic CSVs only, written under ``tmp_path``. No network access, no real
securities, no account credentials, and no personal portfolio information.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pysharpe.data.interop import (
    HoldingsSnapshot,
    Position,
    infer_column_mapping,
    normalise_ticker,
    read_holdings_snapshot,
)
from pysharpe.exceptions import DataIngestionError, DataValidationError
from pysharpe.execution.cash_flow_rebalance import allocate_contribution_cash_flow

AS_OF = date(2025, 3, 31)
FX_RATES = {"USD": 1.35}

# An export using headers from a typical portfolio-tracking system.
VENDOR_CSV = """Instrument,Holding Account,Units Held,Current Price,Currency,Cost Basis
VFV.TO,TFSA-A,10,150.00,CAD,"1,500.00"
QQQ,TAXABLE-B,5,50.00,USD,
"""

MAPPING = {
    "ticker": "Instrument",
    "account": "Holding Account",
    "quantity": "Units Held",
    "price": "Current Price",
    "currency": "Currency",
    "cost_basis": "Cost Basis",
}

REQUIRED_MAPPING = {k: v for k, v in MAPPING.items() if k != "cost_basis"}

# Report-shaped export: market group headers, per-position rows, subtotals, and
# a grand total. Values are synthetic.
REPORT_SHAPED_CSV = """Instrument,Holding Account,Units Held,Current Price,Currency,Value,Capital Gains
NYSE,Taxable,,,,,
EXAMPLE.NYSE,Taxable,2,100.00,USD,"200.00","20.00"
Total,Taxable,,,,"200.00","20.00"
TSE,Taxable,,,,,
EXAMPLE.TSE,Taxable,4,50.00,CAD,"200.00","15.00"
Total,Taxable,,,,"200.00","15.00"
Grand Total,Taxable,,,,"400.00","35.00"
"""


def _write(tmp_path: Path, content: str, name: str = "export.csv") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _snapshot(tmp_path: Path) -> HoldingsSnapshot:
    return read_holdings_snapshot(
        _write(tmp_path, VENDOR_CSV),
        columns=MAPPING,
        as_of=AS_OF,
        fx_rates=FX_RATES,
    )


# -- Parsing and valuation ---------------------------------------------------


def test_reads_export_through_column_mapping(tmp_path):
    snapshot = _snapshot(tmp_path)

    assert snapshot.as_of == AS_OF
    assert len(snapshot.positions) == 2
    assert {(p.account, p.ticker) for p in snapshot.positions} == {
        ("TFSA-A", "VFV.TO"),
        ("TAXABLE-B", "QQQ"),
    }

    vfv, qqq = snapshot.positions
    assert vfv.quantity == 10.0
    assert vfv.price == 150.0
    assert vfv.currency == "CAD"
    assert vfv.cost_basis_cad == 1500.0
    assert qqq.currency == "USD"


def test_cad_valuation_uses_snapshot_price_and_fx(tmp_path):
    snapshot = _snapshot(tmp_path)
    vfv, qqq = snapshot.positions

    assert snapshot.position_value_cad(vfv) == pytest.approx(10 * 150.0)
    assert snapshot.position_value_cad(qqq) == pytest.approx(5 * 50.0 * 1.35)
    assert snapshot.securities_value_cad() == pytest.approx(1500.0 + 337.5)
    assert snapshot.total_value_cad() == pytest.approx(1837.5)


def test_cash_and_total_value(tmp_path):
    snapshot = read_holdings_snapshot(
        _write(tmp_path, VENDOR_CSV),
        columns=MAPPING,
        as_of=AS_OF,
        fx_rates=FX_RATES,
        cash={"TFSA-A": {"CAD": 250.0}, "TAXABLE-B": {"USD": 100.0}},
    )

    assert snapshot.cash_value_cad("TFSA-A") == pytest.approx(250.0)
    assert snapshot.cash_value_cad("TAXABLE-B") == pytest.approx(135.0)
    assert snapshot.cash_value_cad() == pytest.approx(385.0)
    assert snapshot.total_value_cad() == pytest.approx(1500.0 + 337.5 + 385.0)


# -- Downstream contracts ----------------------------------------------------


def test_current_balances_matches_allocator_contract(tmp_path):
    balances = _snapshot(tmp_path).current_balances()

    assert list(balances.columns) == ["ticker", "account", "current_value"]
    rows = balances.set_index(["account", "ticker"])["current_value"]
    assert rows.loc[("TFSA-A", "VFV.TO")] == pytest.approx(1500.0)
    assert rows.loc[("TAXABLE-B", "QQQ")] == pytest.approx(337.5)


def test_holdings_frame_matches_rebalance_contract(tmp_path):
    frame = _snapshot(tmp_path).holdings_frame()

    assert list(frame.columns) == [
        "ticker",
        "account",
        "shares",
        "current_value",
        "currency",
    ]
    row = frame.set_index("ticker").loc["QQQ"]
    assert row["shares"] == 5.0
    assert row["current_value"] == pytest.approx(337.5)
    assert row["currency"] == "USD"


def test_snapshot_feeds_the_contribution_allocator(tmp_path):
    balances = _snapshot(tmp_path).current_balances()

    result = allocate_contribution_cash_flow(
        cash_amount=1000.0,
        current_balances=balances,
        target_weights={"VFV.TO": 0.6, "QQQ": 0.4},
        account_types={"VFV.TO": "TFSA-A", "QQQ": "TAXABLE-B"},
        opportunity_scores={"VFV.TO": 0.5, "QQQ": 0.5},
    )

    buys = result.buy_orders
    assert not buys.empty
    assert set(buys["ticker"]) <= {"VFV.TO", "QQQ"}
    assert buys["buy_amount"].sum() <= 1000.0
    assert result.cash_remaining >= 0.0


# -- Known/unknown cost basis ------------------------------------------------


def test_blank_cost_basis_stays_unknown(tmp_path):
    snapshot = _snapshot(tmp_path)
    qqq = snapshot.positions[1]

    assert qqq.cost_basis_cad is None
    assert qqq.cost_basis_cad != 0.0
    assert snapshot.missing_cost_basis() == (qqq,)


def test_cost_basis_is_optional_in_the_mapping(tmp_path):
    snapshot = read_holdings_snapshot(
        _write(tmp_path, VENDOR_CSV),
        columns=REQUIRED_MAPPING,
        as_of=AS_OF,
        fx_rates=FX_RATES,
    )

    assert all(p.cost_basis_cad is None for p in snapshot.positions)
    assert len(snapshot.missing_cost_basis()) == 2


# -- Explicit failures -------------------------------------------------------


def test_missing_fx_rate_names_the_affected_positions(tmp_path):
    with pytest.raises(DataValidationError, match="QQQ in TAXABLE-B"):
        read_holdings_snapshot(
            _write(tmp_path, VENDOR_CSV),
            columns=MAPPING,
            as_of=AS_OF,
            fx_rates={},
        )


def test_cad_needs_no_fx_rate(tmp_path):
    snapshot = read_holdings_snapshot(
        _write(
            tmp_path,
            "ticker,account,quantity,price,currency\nVFV.TO,TFSA-A,10,150,CAD\n",
        ),
        columns={
            "ticker": "ticker",
            "account": "account",
            "quantity": "quantity",
            "price": "price",
            "currency": "currency",
        },
        as_of=AS_OF,
        fx_rates={},
    )
    assert snapshot.total_value_cad() == pytest.approx(1500.0)


def test_duplicate_account_and_ticker_is_rejected(tmp_path):
    csv = (
        "ticker,account,quantity,price,currency\n"
        "VFV.TO,TFSA-A,10,150,CAD\n"
        "VFV.TO,TFSA-A,5,150,CAD\n"
    )
    with pytest.raises(DataValidationError, match="Duplicate"):
        read_holdings_snapshot(
            _write(tmp_path, csv),
            columns={
                "ticker": "ticker",
                "account": "account",
                "quantity": "quantity",
                "price": "price",
                "currency": "currency",
            },
            as_of=AS_OF,
            fx_rates={},
        )


def test_same_ticker_in_two_accounts_is_allowed(tmp_path):
    csv = (
        "ticker,account,quantity,price,currency\n"
        "VFV.TO,TFSA-A,10,150,CAD\n"
        "VFV.TO,TAXABLE-A,4,150,CAD\n"
    )
    snapshot = read_holdings_snapshot(
        _write(tmp_path, csv),
        columns={
            "ticker": "ticker",
            "account": "account",
            "quantity": "quantity",
            "price": "price",
            "currency": "currency",
        },
        as_of=AS_OF,
        fx_rates={},
    )
    assert snapshot.total_value_cad() == pytest.approx(14 * 150.0)
    assert len(snapshot.current_balances()) == 2


def test_non_positive_quantity_is_rejected(tmp_path):
    csv = "ticker,account,quantity,price,currency\nVFV.TO,TFSA-A,0,150,CAD\n"
    with pytest.raises(DataValidationError, match="strictly positive"):
        read_holdings_snapshot(
            _write(tmp_path, csv),
            columns={
                "ticker": "ticker",
                "account": "account",
                "quantity": "quantity",
                "price": "price",
                "currency": "currency",
            },
            as_of=AS_OF,
            fx_rates={},
        )


def test_non_numeric_quantity_names_the_row(tmp_path):
    csv = "ticker,account,quantity,price,currency\nVFV.TO,TFSA-A,ten,150,CAD\n"
    with pytest.raises(DataValidationError, match=r"Row 1: quantity"):
        read_holdings_snapshot(
            _write(tmp_path, csv),
            columns={
                "ticker": "ticker",
                "account": "account",
                "quantity": "quantity",
                "price": "price",
                "currency": "currency",
            },
            as_of=AS_OF,
            fx_rates={},
        )


def test_missing_required_canonical_column_is_rejected(tmp_path):
    with pytest.raises(DataValidationError, match="Cannot map required canonical"):
        read_holdings_snapshot(
            _write(tmp_path, VENDOR_CSV),
            columns={k: v for k, v in MAPPING.items() if k != "price"},
            as_of=AS_OF,
            fx_rates=FX_RATES,
        )


def test_unknown_canonical_column_is_rejected(tmp_path):
    with pytest.raises(DataValidationError, match="Unknown canonical"):
        read_holdings_snapshot(
            _write(tmp_path, VENDOR_CSV),
            columns={**MAPPING, "market_value": "Value"},
            as_of=AS_OF,
            fx_rates=FX_RATES,
        )


def test_missing_header_is_rejected(tmp_path):
    csv = "Instrument,Units Held,Current Price,Currency\nVFV.TO,10,150,CAD\n"
    with pytest.raises(DataValidationError, match="Mapped columns not found"):
        read_holdings_snapshot(
            _write(tmp_path, csv),
            columns=MAPPING,
            as_of=AS_OF,
            fx_rates=FX_RATES,
        )


def test_blank_rows_are_skipped(tmp_path):
    csv = VENDOR_CSV + ",,,,,\n"
    snapshot = read_holdings_snapshot(
        _write(tmp_path, csv),
        columns=MAPPING,
        as_of=AS_OF,
        fx_rates=FX_RATES,
    )
    assert len(snapshot.positions) == 2


def test_empty_file_is_an_ingestion_error(tmp_path):
    path = _write(tmp_path, "")
    with pytest.raises(DataIngestionError, match="empty"):
        read_holdings_snapshot(path, columns=MAPPING, as_of=AS_OF, fx_rates=FX_RATES)


def test_missing_file_is_an_ingestion_error(tmp_path):
    with pytest.raises(DataIngestionError, match="not found"):
        read_holdings_snapshot(
            tmp_path / "absent.csv",
            columns=MAPPING,
            as_of=AS_OF,
            fx_rates=FX_RATES,
        )


def test_currency_is_case_insensitive_and_defaults_to_cad(tmp_path):
    csv = (
        "ticker,account,quantity,price,currency\n"
        "VFV.TO,TFSA-A,10,150,cad\n"
        "QQQ,TAXABLE-B,5,50,usd\n"
    )
    snapshot = read_holdings_snapshot(
        _write(tmp_path, csv),
        columns={
            "ticker": "ticker",
            "account": "account",
            "quantity": "quantity",
            "price": "price",
            "currency": "currency",
        },
        as_of=AS_OF,
        fx_rates=FX_RATES,
    )
    assert [p.currency for p in snapshot.positions] == ["CAD", "USD"]
    assert snapshot.total_value_cad() == pytest.approx(1500.0 + 337.5)


# -- Header inference and report-shaped exports -----------------------------


def test_infer_column_mapping_from_vendor_headers():
    mapping = infer_column_mapping(
        ["Instrument", "Holding Account", "Units Held", "Current Price", "Currency"]
    )
    assert mapping == REQUIRED_MAPPING


def test_infer_column_mapping_ignores_parenthetical_qualifiers():
    mapping = infer_column_mapping(
        ["Symbol", "Units", "Price (CA$)", "Ccy", "Cost Basis (CA$)"]
    )
    assert mapping["cost_basis"] == "Cost Basis (CA$)"
    assert mapping["currency"] == "Ccy"
    assert mapping["price"] == "Price (CA$)"


def test_infer_column_mapping_rejects_ambiguous_headers():
    with pytest.raises(DataValidationError, match="Multiple columns could be"):
        infer_column_mapping(["Instrument", "Symbol", "Units", "Price", "Currency"])


def test_infer_column_mapping_override_resolves_ambiguity():
    mapping = infer_column_mapping(
        ["Instrument", "Symbol", "Units", "Price", "Currency"],
        overrides={"ticker": "Symbol"},
    )
    assert mapping["ticker"] == "Symbol"


def test_columns_can_be_inferred_at_import(tmp_path):
    snapshot = read_holdings_snapshot(
        _write(tmp_path, VENDOR_CSV), as_of=AS_OF, fx_rates=FX_RATES
    )

    assert len(snapshot.positions) == 2
    assert snapshot.total_value_cad() == pytest.approx(1500.0 + 337.5)
    assert snapshot.positions[0].cost_basis_cad == 1500.0


def test_report_group_and_total_rows_are_skipped(tmp_path):
    snapshot = read_holdings_snapshot(
        _write(tmp_path, REPORT_SHAPED_CSV), as_of=AS_OF, fx_rates=FX_RATES
    )

    assert [p.ticker for p in snapshot.positions] == ["EXAMPLE.NYSE", "EXAMPLE.TSE"]
    assert snapshot.total_value_cad() == pytest.approx(200.0 * 1.35 + 200.0)


def test_row_with_units_but_no_price_is_an_error(tmp_path):
    csv = "ticker,account,quantity,price,currency\nTotal,Taxable,5,,CAD\n"
    with pytest.raises(DataValidationError, match="Row 1: price is empty"):
        read_holdings_snapshot(_write(tmp_path, csv), as_of=AS_OF, fx_rates=FX_RATES)


def test_row_with_numbers_but_no_identity_is_an_error(tmp_path):
    csv = "ticker,account,quantity,price,currency\n,TFSA-A,5,100,CAD\n"
    with pytest.raises(DataValidationError, match="ticker must be non-empty"):
        read_holdings_snapshot(_write(tmp_path, csv), as_of=AS_OF, fx_rates=FX_RATES)


# -- Ticker normalisation ----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("VFV.TSE", "VFV.TO"),
        ("XEQT.TSE", "XEQT.TO"),
        ("VOO.NYSE", "VOO"),
        ("QQQ.NASDAQ", "QQQ"),
        ("VFV | TSE", "VFV.TO"),
        ("VOO | NYSE", "VOO"),
        ("VFV.TO", "VFV.TO"),
        ("VOO", "VOO"),
        ("BRK.B", "BRK.B"),
        ("VFV.TO.TSE", "VFV.TO"),
    ],
)
def test_normalise_ticker(raw, expected):
    assert normalise_ticker(raw) == expected


def test_normalise_ticker_accepts_extra_markets():
    assert normalise_ticker("VOD.LSE", markets={"LSE": ".L"}) == "VOD.L"


def test_normalise_ticker_rejects_empty():
    with pytest.raises(DataValidationError, match="Ticker is empty"):
        normalise_ticker("   ")


def test_normalise_ticker_rejects_market_only_label():
    with pytest.raises(DataValidationError, match="no instrument name"):
        normalise_ticker(".TSE")


def test_import_applies_ticker_transform(tmp_path):
    csv = (
        "Instrument,Holding Account,Units Held,Current Price,Currency\n"
        "VFV.TSE,TFSA-A,10,150.00,CAD\n"
        "VOO.NYSE,TAXABLE-A,5,500.00,USD\n"
    )
    path = _write(tmp_path, csv)

    raw = read_holdings_snapshot(path, as_of=AS_OF, fx_rates=FX_RATES)
    assert [p.ticker for p in raw.positions] == ["VFV.TSE", "VOO.NYSE"]

    normalised = read_holdings_snapshot(
        path, as_of=AS_OF, fx_rates=FX_RATES, ticker_transform=normalise_ticker
    )
    assert [p.ticker for p in normalised.positions] == ["VFV.TO", "VOO"]
    assert normalised.total_value_cad() == pytest.approx(raw.total_value_cad())


def test_duplicate_after_normalisation_is_rejected(tmp_path):
    csv = (
        "ticker,account,quantity,price,currency\n"
        "VFV.TSE,TFSA-A,10,150,CAD\n"
        "VFV.TO,TFSA-A,5,150,CAD\n"
    )
    with pytest.raises(DataValidationError, match="Duplicate"):
        read_holdings_snapshot(
            _write(tmp_path, csv),
            as_of=AS_OF,
            fx_rates=FX_RATES,
            ticker_transform=normalise_ticker,
        )


def test_transform_returning_empty_ticker_is_rejected(tmp_path):
    with pytest.raises(DataValidationError, match="empty ticker"):
        read_holdings_snapshot(
            _write(tmp_path, VENDOR_CSV),
            as_of=AS_OF,
            fx_rates=FX_RATES,
            ticker_transform=lambda _ticker: "  ",
        )


# -- Constructor guards ------------------------------------------------------


def test_non_positive_fx_rate_is_rejected():
    with pytest.raises(DataValidationError, match="strictly positive"):
        HoldingsSnapshot(
            as_of=AS_OF,
            positions=(Position("TFSA-A", "VFV.TO", 1, 100.0),),
            fx_rates={"USD": 0.0},
        )


def test_negative_cash_is_rejected():
    with pytest.raises(DataValidationError, match="non-negative"):
        HoldingsSnapshot(
            as_of=AS_OF,
            positions=(Position("TFSA-A", "VFV.TO", 1, 100.0),),
            cash={"TFSA-A": {"CAD": -1.0}},
        )
