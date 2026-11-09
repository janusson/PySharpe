"""Tests for the portfolio X-ray.

All fixtures are synthetic: fixed quantities, prices, and FX. No network
access and no real portfolio data.
"""

from __future__ import annotations

from datetime import date

import pytest

from pysharpe.analysis.xray import portfolio_xray
from pysharpe.data.interop import (
    HoldingsSnapshot,
    Position,
    normalise_ticker,
    read_holdings_snapshot,
)
from pysharpe.exceptions import DataValidationError
from pysharpe.optimization.tax_location import (
    AssetTaxCharacteristics,
    TaxProfile,
)

AS_OF = date(2025, 3, 31)
FX = {"USD": 1.35}

VOO_CHAR = AssetTaxCharacteristics(
    "VOO",
    dividend_yield=0.013,
    is_us_domiciled=True,
    income_frac_eligible_dividends=0.0,
    income_frac_foreign_income=1.0,
)
VFV_CHAR = AssetTaxCharacteristics(
    "VFV.TO",
    dividend_yield=0.012,
    is_cad_wrapped_us_equity=True,
    income_frac_eligible_dividends=0.0,
    income_frac_foreign_income=1.0,
)


def _position(
    account: str, ticker: str, quantity: float, price: float, currency: str = "CAD"
) -> Position:
    return Position(
        account=account,
        ticker=ticker,
        quantity=quantity,
        price=price,
        currency=currency,
    )


def _snapshot(*positions, cash=None) -> HoldingsSnapshot:
    return HoldingsSnapshot(
        as_of=AS_OF,
        positions=tuple(positions),
        cash=cash or {},
        fx_rates=FX,
    )


# -- Weights and concentration ----------------------------------------------


def test_weights_and_effective_holdings():
    report = portfolio_xray(
        _snapshot(
            _position("TFSA-A", "AAA.TO", 90, 10.0),
            _position("TFSA-A", "BBB.TO", 10, 10.0),
        ),
        account_types={"TFSA-A": "TFSA"},
    )

    weights = dict(zip(report.positions["ticker"], report.positions["weight"]))
    assert weights["AAA.TO"] == pytest.approx(0.9)
    assert weights["BBB.TO"] == pytest.approx(0.1)
    assert report.largest_position_weight == pytest.approx(0.9)
    # HHI = 0.9^2 + 0.1^2 = 0.82 -> 1.2195 effective holdings
    assert report.effective_holdings == pytest.approx(1 / 0.82)


def test_concentration_flag_above_threshold():
    report = portfolio_xray(
        _snapshot(
            _position("TFSA-A", "AAA.TO", 90, 10.0),
            _position("TFSA-A", "BBB.TO", 10, 10.0),
        ),
        account_types={"TFSA-A": "TFSA"},
    )

    flags = [f for f in report.flags if f.code == "concentration"]
    assert len(flags) == 1
    assert flags[0].severity == "warning"
    assert "AAA.TO" in flags[0].message


def test_no_concentration_flag_below_threshold():
    report = portfolio_xray(
        _snapshot(
            _position("TFSA-A", "AAA.TO", 25, 10.0),
            _position("TFSA-A", "BBB.TO", 25, 10.0),
            _position("TFSA-A", "CCC.TO", 25, 10.0),
            _position("TFSA-A", "DDD.TO", 25, 10.0),
        ),
        account_types={"TFSA-A": "TFSA"},
    )
    assert [f for f in report.flags if f.code == "concentration"] == []
    assert report.effective_holdings == pytest.approx(4.0)


def test_threshold_must_be_a_fraction():
    with pytest.raises(DataValidationError, match="concentration_threshold"):
        portfolio_xray(
            _snapshot(_position("TFSA-A", "AAA.TO", 1, 10.0)),
            account_types={"TFSA-A": "TFSA"},
            concentration_threshold=25.0,
        )


# -- Currency exposure --------------------------------------------------------


def test_currency_exposure_includes_cash():
    report = portfolio_xray(
        _snapshot(
            _position("TFSA-A", "AAA.TO", 10, 100.0),
            _position("TFSA-A", "VOO", 10, 100.0, "USD"),
            cash={"TFSA-A": {"CAD": 250.0, "USD": 100.0}},
        ),
        account_types={"TFSA-A": "TFSA"},
    )

    exposure = report.currency_exposure.set_index("currency")
    assert exposure.loc["CAD", "value_cad"] == pytest.approx(1000.0 + 250.0)
    assert exposure.loc["USD", "value_cad"] == pytest.approx(1350.0 + 135.0)
    assert report.total_value_cad == pytest.approx(1000.0 + 1350.0 + 250.0 + 135.0)
    assert report.cash_value_cad == pytest.approx(385.0)
    assert exposure["weight"].sum() == pytest.approx(1.0)


# -- Identical exposure -------------------------------------------------------


def test_identical_exposure_across_accounts_is_flagged():
    report = portfolio_xray(
        _snapshot(
            _position("TFSA-A", "VFV.TO", 10, 100.0),
            _position("TAXABLE-A", "VOO", 10, 100.0, "USD"),
        ),
        account_types={"TFSA-A": "TFSA", "TAXABLE-A": "NON_REG"},
    )

    flags = [f for f in report.flags if f.code == "identical_exposure"]
    assert len(flags) == 1
    assert set(flags[0].tickers) == {"VFV.TO", "VOO"}
    assert "same underlying exposure" in flags[0].message
    assert report.identical_groups == ((("TAXABLE-A", "VOO"), ("TFSA-A", "VFV.TO")),)


def test_same_security_in_two_accounts_is_flagged():
    report = portfolio_xray(
        _snapshot(
            _position("TFSA-A", "VFV.TO", 10, 100.0),
            _position("TAXABLE-A", "VFV.TO", 5, 100.0),
        ),
        account_types={"TFSA-A": "TFSA", "TAXABLE-A": "NON_REG"},
    )

    flags = [f for f in report.flags if f.code == "identical_exposure"]
    assert len(flags) == 1
    assert "superficial-loss" in flags[0].message


def test_unrelated_tickers_are_not_grouped():
    report = portfolio_xray(
        _snapshot(
            _position("TFSA-A", "VFV.TO", 10, 100.0),
            _position("TAXABLE-A", "VDY.TO", 10, 100.0),
        ),
        account_types={"TFSA-A": "TFSA", "TAXABLE-A": "NON_REG"},
    )
    assert [f for f in report.flags if f.code == "identical_exposure"] == []
    assert report.identical_groups == ()


def test_normalised_export_clusters_as_identical_property(tmp_path):
    """An export with market codes must reach the same finding as clean tickers."""
    csv = (
        "Instrument,Holding Account,Units Held,Current Price,Currency\n"
        "VFV.TSE,TFSA-A,10,150.00,CAD\n"
        "VOO.NYSE,TAXABLE-A,5,500.00,USD\n"
    )
    path = tmp_path / "export.csv"
    path.write_text(csv, encoding="utf-8")

    snapshot = read_holdings_snapshot(
        path, as_of=AS_OF, fx_rates=FX, ticker_transform=normalise_ticker
    )
    report = portfolio_xray(
        snapshot,
        account_types={"TFSA-A": "TFSA", "TAXABLE-A": "NON_REG"},
    )

    flags = [f for f in report.flags if f.code == "identical_exposure"]
    assert len(flags) == 1
    assert set(flags[0].tickers) == {"VFV.TO", "VOO"}


def test_custom_identical_map_is_honoured():
    report = portfolio_xray(
        _snapshot(
            _position("TFSA-A", "HOUSE-A", 10, 100.0),
            _position("TFSA-A", "HOUSE-B", 10, 50.0),
        ),
        account_types={"TFSA-A": "TFSA"},
        identical_map={"HOUSE-A": frozenset({"HOUSE-B"})},
    )
    flags = [f for f in report.flags if f.code == "identical_exposure"]
    assert len(flags) == 1
    assert set(flags[0].tickers) == {"HOUSE-A", "HOUSE-B"}


# -- Tax location -------------------------------------------------------------


def test_tfsa_withholding_drag_in_basis_points():
    report = portfolio_xray(
        _snapshot(_position("TFSA-A", "VOO", 10, 100.0, "USD")),
        account_types={"TFSA-A": "TFSA"},
        asset_characteristics={"VOO": VOO_CHAR},
    )

    flags = [f for f in report.flags if f.code == "withholding_drag"]
    assert len(flags) == 1
    # 1.3% yield x 15% tax = 0.195% = 19.5 bps
    assert flags[0].drag_bps == pytest.approx(19.5)
    assert "no treaty" in flags[0].message


def test_rrsp_holding_has_no_withholding_drag():
    report = portfolio_xray(
        _snapshot(_position("RRSP-A", "VOO", 10, 100.0, "USD")),
        account_types={"RRSP-A": "RRSP"},
        asset_characteristics={"VOO": VOO_CHAR},
    )
    assert [f for f in report.flags if f.code == "withholding_drag"] == []


def test_cad_wrapped_us_equity_drags_in_tfsa():
    report = portfolio_xray(
        _snapshot(_position("TFSA-A", "VFV.TO", 10, 100.0)),
        account_types={"TFSA-A": "TFSA"},
        asset_characteristics={"VFV.TO": VFV_CHAR},
    )

    flags = [f for f in report.flags if f.code == "withholding_drag"]
    assert len(flags) == 1
    # 1.2% yield x 15% = 18 bps
    assert flags[0].drag_bps == pytest.approx(18.0)


def test_direct_us_holding_points_at_treaty_account_when_present():
    report = portfolio_xray(
        _snapshot(
            _position("TFSA-A", "VOO", 10, 100.0, "USD"),
            _position("RRSP-A", "VFV.TO", 10, 100.0),
        ),
        account_types={"TFSA-A": "TFSA", "RRSP-A": "RRSP"},
        asset_characteristics={"VOO": VOO_CHAR, "VFV.TO": VFV_CHAR},
    )

    withholding = [f for f in report.flags if f.code == "withholding_drag"]
    # VOO in the TFSA (19.5 bps) and VFV.TO in the RRSP (18 bps: the FWT is
    # withheld inside the fund, so treaty protection does not apply).
    assert len(withholding) == 2
    voo_flag = next(f for f in withholding if f.tickers == ("VOO",))
    assert "RRSP-A" in voo_flag.message
    vfv_flag = next(f for f in withholding if f.tickers == ("VFV.TO",))
    assert vfv_flag.drag_bps == pytest.approx(18.0)
    assert "withheld inside the fund" in vfv_flag.message


def test_non_registered_income_drag_is_flagged():
    report = portfolio_xray(
        _snapshot(_position("TAXABLE-A", "VOO", 10, 100.0, "USD")),
        account_types={"TAXABLE-A": "NON_REG"},
        asset_characteristics={"VOO": VOO_CHAR},
        tax_profile=TaxProfile(marginal_tax_rate=0.30),
    )

    flags = [f for f in report.flags if f.code == "income_tax_drag"]
    assert len(flags) == 1
    # 100% foreign income x 30% marginal rate = 3000 bps of the income component
    assert flags[0].drag_bps == pytest.approx(3000.0)
    assert report.marginal_tax_rate == pytest.approx(0.30)


def test_unassessed_assets_are_reported_not_assumed():
    report = portfolio_xray(
        _snapshot(
            _position("TFSA-A", "AAA.TO", 10, 100.0),
            _position("TAXABLE-A", "VOO", 10, 100.0, "USD"),
        ),
        account_types={"TFSA-A": "TFSA", "TAXABLE-A": "NON_REG"},
        asset_characteristics={"VOO": VOO_CHAR},
    )

    flags = [f for f in report.flags if f.code == "unassessed_asset"]
    assert len(flags) == 1
    assert flags[0].tickers == ("AAA.TO",)
    assert "unknown, not" in flags[0].message


# -- Account typing -----------------------------------------------------------


def test_missing_account_type_is_an_error():
    with pytest.raises(DataValidationError, match="missing accounts"):
        portfolio_xray(
            _snapshot(
                _position("TFSA-A", "AAA.TO", 1, 10.0),
                _position("TAXABLE-A", "BBB.TO", 1, 10.0),
            ),
            account_types={"TFSA-A": "TFSA"},
        )


def test_unknown_account_type_is_an_error():
    with pytest.raises(DataValidationError, match="Unknown account type"):
        portfolio_xray(
            _snapshot(_position("MYSTERY", "AAA.TO", 1, 10.0)),
            account_types={"MYSTERY": "PENSION"},
        )


def test_taxable_alias_is_accepted():
    report = portfolio_xray(
        _snapshot(_position("TAXABLE-A", "AAA.TO", 1, 10.0)),
        account_types={"TAXABLE-A": "taxable"},
    )
    assert report.total_value_cad == pytest.approx(10.0)


# -- Edge cases and rendering -------------------------------------------------


def test_cash_only_snapshot():
    report = portfolio_xray(
        _snapshot(cash={"TFSA-A": {"CAD": 500.0}}),
        account_types={"TFSA-A": "TFSA"},
    )

    assert report.positions.empty
    assert report.largest_position_weight == 0.0
    assert report.effective_holdings == 0.0
    assert report.total_value_cad == pytest.approx(500.0)
    assert [f for f in report.flags if f.code == "concentration"] == []


def test_report_and_frame_are_deterministic():
    snapshot = _snapshot(
        _position("TFSA-A", "VFV.TO", 10, 100.0),
        _position("TAXABLE-A", "VOO", 10, 100.0, "USD"),
        cash={"TFSA-A": {"CAD": 50.0}},
    )
    kwargs = {
        "account_types": {"TFSA-A": "TFSA", "TAXABLE-A": "NON_REG"},
        "asset_characteristics": {"VOO": VOO_CHAR, "VFV.TO": VFV_CHAR},
    }
    first = portfolio_xray(snapshot, **kwargs)
    second = portfolio_xray(snapshot, **kwargs)

    assert first.report() == second.report()
    assert first.to_frame().equals(second.to_frame())

    frame = first.to_frame()
    assert list(frame.columns) == [
        "code",
        "severity",
        "tickers",
        "accounts",
        "drag_bps",
        "message",
    ]
    assert frame["severity"].tolist()[0] == "warning"
    report = first.report()
    for section in ("PORTFOLIO X-RAY", "CURRENCY EXPOSURE", "POSITIONS", "FINDINGS"):
        assert section in report
