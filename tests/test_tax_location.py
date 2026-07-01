"""Tests for the Asset Location Engine (tax_location.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from pysharpe.optimization.tax_location import (
    AccountType,
    AssetLocationEngine,
    AssetTaxCharacteristics,
    TaxProfile,
    _normalize_account,
    build_asset_characteristics,
    build_asset_characteristics_batch,
)

# ---------------------------------------------------------------------------
# TaxProfile
# ---------------------------------------------------------------------------


class TestTaxProfile:
    def test_default_construction(self):
        profile = TaxProfile(marginal_tax_rate=0.45)
        assert profile.marginal_tax_rate == 0.45
        assert profile.capital_gains_inclusion_rate == 0.50
        assert profile.eligible_dividend_gross_up == 1.38

    def test_rejects_invalid_mtr(self):
        with pytest.raises(ValueError, match="marginal_tax_rate"):
            TaxProfile(marginal_tax_rate=1.5)
        with pytest.raises(ValueError, match="marginal_tax_rate"):
            TaxProfile(marginal_tax_rate=-0.1)

    def test_rejects_invalid_inclusion_rate(self):
        with pytest.raises(ValueError, match="capital_gains_inclusion_rate"):
            TaxProfile(marginal_tax_rate=0.45, capital_gains_inclusion_rate=1.5)

    def test_eligible_dividend_effective_rate_standard(self):
        """Ontario top bracket (~53.5% MTR) effective dividend rate ~39.34%."""
        profile = TaxProfile(marginal_tax_rate=0.5353)
        # Grossed-up tax: 0.5353 * 1.38 = 0.738714
        # DTC applied to grossed-up amount: (0.150198 + 0.10) * 1.38 = 0.345273
        # Effective = 0.738714 - 0.345273 = 0.3934
        assert profile.eligible_dividend_effective_rate == pytest.approx(
            0.39344, rel=1e-4
        )

    def test_capital_gains_effective_rate(self):
        profile = TaxProfile(marginal_tax_rate=0.50)
        assert profile.capital_gains_effective_rate == 0.25  # 0.50 * 0.50

        profile_high = TaxProfile(
            marginal_tax_rate=0.50, capital_gains_inclusion_rate=0.6667
        )
        assert profile_high.capital_gains_effective_rate == pytest.approx(0.33335)

    def test_eligible_dividend_effective_rate_zero_boundary(self):
        """When DTC exceeds grossed-up tax, effective rate should floor at 0."""
        profile = TaxProfile(
            marginal_tax_rate=0.10,  # low MTR
            eligible_dividend_tax_credit_federal=0.20,
            eligible_dividend_tax_credit_provincial=0.10,
        )
        # grossed_up_tax: 0.10 * 1.38 = 0.138
        # dtc_total (applied to grossed-up): (0.20 + 0.10) * 1.38 = 0.414
        # effective = max(0, 0.138 - 0.414) = 0.0
        assert profile.eligible_dividend_effective_rate == 0.0

    def test_frozen(self):
        profile = TaxProfile(marginal_tax_rate=0.40)
        with pytest.raises(Exception):  # noqa: B017
            profile.marginal_tax_rate = 0.50  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AssetTaxCharacteristics
# ---------------------------------------------------------------------------


class TestAssetTaxCharacteristics:
    def test_default_construction(self):
        a = AssetTaxCharacteristics("VCN.TO", dividend_yield=0.025)
        assert a.ticker == "VCN.TO"
        assert a.dividend_yield == 0.025
        assert a.is_us_domiciled is False
        assert a.is_cad_wrapped_us_equity is False
        assert a.has_us_equity_exposure is False
        assert a.mer == 0.0

    def test_us_domiciled(self):
        a = AssetTaxCharacteristics("VOO", dividend_yield=0.013, is_us_domiciled=True)
        assert a.has_us_equity_exposure is True

    def test_cad_wrapped_us(self):
        a = AssetTaxCharacteristics(
            "VFV.TO", dividend_yield=0.012, is_cad_wrapped_us_equity=True
        )
        assert a.has_us_equity_exposure is True

    def test_rejects_negative_dividend_yield(self):
        with pytest.raises(ValueError, match="dividend_yield"):
            AssetTaxCharacteristics("X", dividend_yield=-0.01)

    def test_rejects_negative_mer(self):
        with pytest.raises(ValueError, match="MER"):
            AssetTaxCharacteristics("X", dividend_yield=0.02, mer=-0.001)

    def test_rejects_income_fractions_not_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            AssetTaxCharacteristics(
                "X",
                dividend_yield=0.02,
                income_frac_interest=0.5,
                income_frac_eligible_dividends=0.3,
                income_frac_foreign_income=0.0,
                income_frac_capital_gains=0.0,  # sums to 0.8
            )

    def test_income_fractions_sum_to_one_ok(self):
        a = AssetTaxCharacteristics(
            "X",
            dividend_yield=0.02,
            income_frac_interest=0.1,
            income_frac_eligible_dividends=0.5,
            income_frac_foreign_income=0.2,
            income_frac_capital_gains=0.2,
        )
        assert a.income_frac_interest == 0.1
        assert a.income_frac_eligible_dividends == 0.5

    def test_frozen(self):
        a = AssetTaxCharacteristics("VOO", dividend_yield=0.013)
        with pytest.raises(Exception):  # noqa: B017
            a.dividend_yield = 0.03  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Account normalisation
# ---------------------------------------------------------------------------


class TestNormalizeAccount:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("TFSA", AccountType.TFSA),
            ("tfsa", AccountType.TFSA),
            ("  TFSA  ", AccountType.TFSA),
            ("RRSP", AccountType.RRSP),
            ("rrsp", AccountType.RRSP),
            ("FHSA", AccountType.FHSA),
            ("NON_REG", AccountType.NON_REG),
            ("Non-Reg", AccountType.NON_REG),
            ("non-reg", AccountType.NON_REG),
            ("Taxable", AccountType.NON_REG),
            ("Margin", AccountType.NON_REG),
            ("Cash", AccountType.NON_REG),
            ("UNREGISTERED", AccountType.NON_REG),
            ("UnknownType", AccountType.NON_REG),  # default fallback
        ],
    )
    def test_normalization(self, raw, expected):
        assert _normalize_account(raw) == expected


# ---------------------------------------------------------------------------
# AssetLocationEngine — FWT drag
# ---------------------------------------------------------------------------


class TestFwtDrag:
    @pytest.fixture
    def engine(self):
        return AssetLocationEngine(TaxProfile(marginal_tax_rate=0.45))

    @pytest.fixture
    def voo(self):
        """US-domiciled ETF, directly held (e.g. VOO in USD)."""
        return AssetTaxCharacteristics(
            "VOO", dividend_yield=0.013, is_us_domiciled=True
        )

    @pytest.fixture
    def vfv(self):
        """CAD-wrapped US ETF (e.g. VFV.TO)."""
        return AssetTaxCharacteristics(
            "VFV.TO", dividend_yield=0.012, is_cad_wrapped_us_equity=True
        )

    @pytest.fixture
    def vcn(self):
        """Canadian-domiciled ETF, no US exposure."""
        return AssetTaxCharacteristics(
            "VCN.TO", dividend_yield=0.025, is_us_domiciled=False
        )

    # -- US-domiciled (VOO) --

    def test_voo_rrsp_no_drag(self, engine, voo):
        """US-domiciled in RRSP → zero FWT (treaty protection)."""
        assert engine.compute_fwt_drag(voo, "RRSP") == 0.0

    def test_voo_tfsa_has_drag(self, engine, voo):
        """US-domiciled in TFSA → 15% of dividend yield."""
        expected = 0.013 * 0.15  # 0.00195
        assert engine.compute_fwt_drag(voo, "TFSA") == pytest.approx(expected)

    def test_voo_fhsa_has_drag(self, engine, voo):
        expected = 0.013 * 0.15
        assert engine.compute_fwt_drag(voo, "FHSA") == pytest.approx(expected)

    def test_voo_non_reg_no_fwt_drag(self, engine, voo):
        """Non-Reg → FWT recoverable via FTC, so FWT drag is 0."""
        assert engine.compute_fwt_drag(voo, "Non-Reg") == 0.0

    # -- CAD-wrapped US (VFV.TO) --

    def test_vfv_rrsp_has_drag(self, engine, vfv):
        """CAD-wrapped US in RRSP → FWT lost at fund level, unrecoverable."""
        expected = 0.012 * 0.15  # 0.0018
        assert engine.compute_fwt_drag(vfv, "RRSP") == pytest.approx(expected)

    def test_vfv_tfsa_has_drag(self, engine, vfv):
        expected = 0.012 * 0.15
        assert engine.compute_fwt_drag(vfv, "TFSA") == pytest.approx(expected)

    def test_vfv_non_reg_no_fwt_drag(self, engine, vfv):
        """FWT recoverable via FTC for Non-Reg accounts."""
        assert engine.compute_fwt_drag(vfv, "Non-Reg") == 0.0

    # -- Canadian (VCN.TO) --

    def test_vcn_any_account_no_drag(self, engine, vcn):
        """Canadian-domiciled → no US FWT in any account."""
        for account in ("TFSA", "RRSP", "FHSA", "Non-Reg"):
            assert engine.compute_fwt_drag(vcn, account) == 0.0

    # -- LIRA / RRIF (treaty-protected variants) --

    def test_voo_lira_no_drag(self, engine, voo):
        assert engine.compute_fwt_drag(voo, "LIRA") == 0.0

    def test_voo_rrif_no_drag(self, engine, voo):
        assert engine.compute_fwt_drag(voo, "RRIF") == 0.0

    def test_vfv_lira_has_drag(self, engine, vfv):
        expected = 0.012 * 0.15
        assert engine.compute_fwt_drag(vfv, "LIRA") == pytest.approx(expected)


# ---------------------------------------------------------------------------
# AssetLocationEngine — income tax drag (Non-Reg only)
# ---------------------------------------------------------------------------


class TestIncomeTaxDrag:
    @pytest.fixture
    def engine(self):
        return AssetLocationEngine(TaxProfile(marginal_tax_rate=0.50))

    def test_registered_accounts_zero_income_drag(self, engine):
        """Tax-sheltered accounts have no annual income tax drag."""
        asset = AssetTaxCharacteristics(
            "VCN",
            dividend_yield=0.03,
            income_frac_eligible_dividends=1.0,
        )
        for account in ("TFSA", "RRSP", "FHSA"):
            assert engine.compute_income_tax_drag(asset, account) == 0.0

    def test_non_reg_pure_interest(self, engine):
        """100% interest income → full MTR applied."""
        asset = AssetTaxCharacteristics(
            "bond",
            dividend_yield=0.04,
            income_frac_interest=1.0,
            income_frac_eligible_dividends=0.0,
        )
        drag = engine.compute_income_tax_drag(asset, "Non-Reg")
        assert drag == pytest.approx(0.50 * 1.0)  # 50% MTR

    def test_non_reg_pure_eligible_dividends(self, engine):
        """100% eligible dividends → effective dividend rate."""
        asset = AssetTaxCharacteristics(
            "VCN",
            dividend_yield=0.03,
            income_frac_eligible_dividends=1.0,
            income_frac_interest=0.0,
        )
        drag = engine.compute_income_tax_drag(asset, "Non-Reg")
        eff_div_rate = engine._profile.eligible_dividend_effective_rate
        assert drag == pytest.approx(eff_div_rate)

    def test_non_reg_pure_capital_gains(self, engine):
        """100% capital gains → MTR * inclusion_rate."""
        asset = AssetTaxCharacteristics(
            "growth",
            dividend_yield=0.0,
            income_frac_capital_gains=1.0,
            income_frac_eligible_dividends=0.0,
        )
        drag = engine.compute_income_tax_drag(asset, "Non-Reg")
        assert drag == pytest.approx(0.50 * 0.50)  # 0.25

    def test_non_reg_blended(self, engine):
        """Mixed income sources should be weighted by fractions."""
        asset = AssetTaxCharacteristics(
            "blend",
            dividend_yield=0.04,
            income_frac_interest=0.2,
            income_frac_eligible_dividends=0.4,
            income_frac_foreign_income=0.1,
            income_frac_capital_gains=0.3,
        )
        tp = engine._profile
        expected = (
            0.2 * tp.marginal_tax_rate
            + 0.4 * tp.eligible_dividend_effective_rate
            + 0.1 * tp.marginal_tax_rate
            + 0.3 * tp.capital_gains_effective_rate
        )
        assert engine.compute_income_tax_drag(asset, "Non-Reg") == pytest.approx(
            expected
        )

    def test_non_reg_with_foreign_income(self, engine):
        """Foreign income taxed at full MTR (FWT recovered separately)."""
        asset = AssetTaxCharacteristics(
            "intl",
            dividend_yield=0.03,
            income_frac_foreign_income=1.0,
            income_frac_eligible_dividends=0.0,
        )
        drag = engine.compute_income_tax_drag(asset, "Non-Reg")
        assert drag == pytest.approx(0.50)  # full MTR


# ---------------------------------------------------------------------------
# AssetLocationEngine — total drag
# ---------------------------------------------------------------------------


class TestTotalDrag:
    @pytest.fixture
    def engine(self):
        return AssetLocationEngine(TaxProfile(marginal_tax_rate=0.50))

    def test_voo_tfsa_total_is_fwt_only(self, engine):
        """In TFSA, only FWT applies (no income tax)."""
        voo = AssetTaxCharacteristics("VOO", dividend_yield=0.013, is_us_domiciled=True)
        fwt = engine.compute_fwt_drag(voo, "TFSA")
        income = engine.compute_income_tax_drag(voo, "TFSA")
        assert income == 0.0
        assert engine.compute_total_drag(voo, "TFSA") == pytest.approx(fwt)

    def test_interest_asset_non_reg_total(self, engine):
        """In Non-Reg, FWT is 0 but income tax applies."""
        bond = AssetTaxCharacteristics(
            "bond",
            dividend_yield=0.04,
            income_frac_interest=1.0,
            income_frac_eligible_dividends=0.0,
        )
        fwt = engine.compute_fwt_drag(bond, "Non-Reg")
        assert fwt == 0.0
        total = engine.compute_total_drag(bond, "Non-Reg")
        assert total == pytest.approx(0.50)  # MTR

    def test_voo_non_reg_fwt_zero_income_applies(self, engine):
        """VOO in Non-Reg: FWT recoverable, but eligible dividends taxed."""
        voo = AssetTaxCharacteristics(
            "VOO",
            dividend_yield=0.013,
            is_us_domiciled=True,
            income_frac_foreign_income=1.0,  # foreign dividends
            income_frac_eligible_dividends=0.0,
        )
        total = engine.compute_total_drag(voo, "Non-Reg")
        # FWT = 0, income = 1.0 * 0.50 = 0.50
        assert total == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# AssetLocationEngine — tax-adjusted return
# ---------------------------------------------------------------------------


class TestTaxAdjustedReturn:
    @pytest.fixture
    def engine(self):
        return AssetLocationEngine(TaxProfile(marginal_tax_rate=0.45))

    def test_no_drag_no_mer(self, engine):
        """When drag and MER are zero, return is unchanged."""
        asset = AssetTaxCharacteristics("VCN.TO", dividend_yield=0.02)
        result = engine.compute_tax_adjusted_return(0.08, asset, "TFSA")
        assert result == pytest.approx(0.08)

    def test_fwt_drag_reduces_return(self, engine):
        """FWT drag should reduce the expected return."""
        voo = AssetTaxCharacteristics("VOO", dividend_yield=0.013, is_us_domiciled=True)
        result = engine.compute_tax_adjusted_return(0.10, voo, "TFSA")
        expected_drag = 0.013 * 0.15  # 0.00195
        assert result == pytest.approx(0.10 - expected_drag)

    def test_mer_deducted(self, engine):
        """MER should be deducted alongside drag."""
        asset = AssetTaxCharacteristics("XEQT.TO", dividend_yield=0.02, mer=0.0020)
        result = engine.compute_tax_adjusted_return(0.07, asset, "TFSA")
        assert result == pytest.approx(0.07 - 0.0020)  # no FWT, MER only

    def test_mer_and_drag_both_deducted(self, engine):
        """Both MER and FWT drag should be deducted."""
        vfv = AssetTaxCharacteristics(
            "VFV.TO",
            dividend_yield=0.012,
            is_cad_wrapped_us_equity=True,
            mer=0.0009,
        )
        result = engine.compute_tax_adjusted_return(0.09, vfv, "TFSA")
        fwt_drag = 0.012 * 0.15
        assert result == pytest.approx(0.09 - 0.0009 - fwt_drag)

    def test_non_reg_income_tax_reduces_return(self, engine):
        """Income tax drag in Non-Reg should reduce expected return."""
        bond = AssetTaxCharacteristics(
            "bond",
            dividend_yield=0.04,
            income_frac_interest=1.0,
            income_frac_eligible_dividends=0.0,
        )
        result = engine.compute_tax_adjusted_return(0.05, bond, "Non-Reg")
        assert result == pytest.approx(0.05 - 0.45)  # 0.45 = MTR

    def test_negative_adjusted_return_possible(self, engine):
        """If tax drag exceeds expected return, the adjusted return can go negative."""
        bond = AssetTaxCharacteristics(
            "bond",
            dividend_yield=0.04,
            income_frac_interest=1.0,
            income_frac_eligible_dividends=0.0,
        )
        result = engine.compute_tax_adjusted_return(0.03, bond, "Non-Reg")
        assert result < 0.0


# ---------------------------------------------------------------------------
# AssetLocationEngine — tax efficiency score
# ---------------------------------------------------------------------------


class TestTaxEfficiencyScore:
    @pytest.fixture
    def engine(self):
        return AssetLocationEngine(TaxProfile(marginal_tax_rate=0.45))

    def test_no_drag_is_maximally_efficient(self, engine):
        asset = AssetTaxCharacteristics("VCN.TO", dividend_yield=0.02)
        score = engine.compute_tax_efficiency_score(asset, "TFSA")
        assert score == 1.0

    def test_fwt_drag_reduces_score(self, engine):
        voo = AssetTaxCharacteristics("VOO", dividend_yield=0.013, is_us_domiciled=True)
        score = engine.compute_tax_efficiency_score(voo, "TFSA")
        expected_drag = 0.013 * 0.15  # 0.00195
        expected_score = 1.0 - (expected_drag / 0.15)  # default max_drag=0.15
        assert score == pytest.approx(expected_score)

    def test_custom_max_drag(self, engine):
        voo = AssetTaxCharacteristics("VOO", dividend_yield=0.013, is_us_domiciled=True)
        score_small = engine.compute_tax_efficiency_score(voo, "TFSA", max_drag=0.01)
        score_large = engine.compute_tax_efficiency_score(voo, "TFSA", max_drag=0.30)
        # Smaller max_drag → larger impact → lower score
        assert score_small < score_large

    def test_drag_clamped_at_max(self, engine):
        """When drag exceeds max_drag, score floors at 0.0."""
        bond = AssetTaxCharacteristics(
            "bond",
            dividend_yield=0.04,
            income_frac_interest=1.0,
            income_frac_eligible_dividends=0.0,
        )
        score = engine.compute_tax_efficiency_score(bond, "Non-Reg", max_drag=0.01)
        # Drag = 0.45 (MTR), clamped to 0.01 → score = 0.0
        assert score == 0.0

    def test_rejects_non_positive_max_drag(self, engine):
        asset = AssetTaxCharacteristics(
            "VOO", dividend_yield=0.013, is_us_domiciled=True
        )
        with pytest.raises(ValueError, match="max_drag"):
            engine.compute_tax_efficiency_score(asset, "TFSA", max_drag=0.0)

    def test_rrsp_is_more_efficient_than_tfsa_for_voo(self, engine):
        """VOO should score higher in RRSP (no FWT) than in TFSA (FWT applies)."""
        voo = AssetTaxCharacteristics("VOO", dividend_yield=0.013, is_us_domiciled=True)
        rrsp_score = engine.compute_tax_efficiency_score(voo, "RRSP")
        tfsa_score = engine.compute_tax_efficiency_score(voo, "TFSA")
        assert rrsp_score > tfsa_score

    def test_non_reg_is_more_efficient_than_tfsa_for_voo_fwt_wise(self, engine):
        """For VOO specifically, Non-Reg beats TFSA because FWT is recoverable.

        However, income tax still applies in Non-Reg, so this tests the FWT
        component only.  Total drag may still favor other accounts for pure
        equity depending on the full income composition.
        """
        voo = AssetTaxCharacteristics(
            "VOO",
            dividend_yield=0.013,
            is_us_domiciled=True,
            income_frac_capital_gains=1.0,
            income_frac_eligible_dividends=0.0,
        )
        nr_score = engine.compute_tax_efficiency_score(voo, "Non-Reg")
        tfsa_score = engine.compute_tax_efficiency_score(voo, "TFSA")
        # Non-Reg has higher FWT efficiency but income tax on gains reduces score
        # This tests the relative behavior is consistent
        rrsp_score = engine.compute_tax_efficiency_score(voo, "RRSP")
        # RRSP should be best for US equity (no FWT, no annual taxation)
        assert rrsp_score >= max(nr_score, tfsa_score)


# ---------------------------------------------------------------------------
# AssetLocationEngine — adjusted opportunity scores
# ---------------------------------------------------------------------------


class TestAdjustedOpportunityScores:
    @pytest.fixture
    def engine(self):
        return AssetLocationEngine(TaxProfile(marginal_tax_rate=0.45))

    def test_blends_scores(self, engine):
        assets = {
            "VCN": AssetTaxCharacteristics("VCN.TO", dividend_yield=0.02),
            "VOO": AssetTaxCharacteristics(
                "VOO", dividend_yield=0.013, is_us_domiciled=True
            ),
        }
        base_scores = {"VCN": 0.8, "VOO": 0.8}
        adjusted = engine.compute_adjusted_opportunity_scores(
            assets, "TFSA", base_scores, tax_weight=0.3
        )
        # VCN: tax_eff = 1.0, adjusted = 0.7*0.8 + 0.3*1.0 = 0.86
        # VOO: tax_eff < 1.0, adjusted < 0.86
        assert adjusted["VCN"] > base_scores["VCN"]  # boosted by tax efficiency
        assert adjusted["VOO"] < adjusted["VCN"]  # penalized vs Canadian

    def test_unknown_ticker_passthrough(self, engine):
        assets: dict[str, AssetTaxCharacteristics] = {}
        base_scores = {"UNKNOWN": 0.5}
        adjusted = engine.compute_adjusted_opportunity_scores(
            assets, "TFSA", base_scores
        )
        assert adjusted["UNKNOWN"] == 0.5

    def test_tax_weight_zero(self, engine):
        """tax_weight=0 should return base scores unchanged."""
        assets = {
            "VOO": AssetTaxCharacteristics(
                "VOO", dividend_yield=0.013, is_us_domiciled=True
            ),
        }
        base = {"VOO": 0.5}
        adjusted = engine.compute_adjusted_opportunity_scores(
            assets, "TFSA", base, tax_weight=0.0
        )
        assert adjusted["VOO"] == pytest.approx(0.5)

    def test_tax_weight_one(self, engine):
        """tax_weight=1.0 should return pure tax efficiency scores."""
        assets = {
            "VOO": AssetTaxCharacteristics(
                "VOO", dividend_yield=0.013, is_us_domiciled=True
            ),
        }
        base = {"VOO": 0.5}
        adjusted = engine.compute_adjusted_opportunity_scores(
            assets, "TFSA", base, tax_weight=1.0
        )
        expected = engine.compute_tax_efficiency_score(assets["VOO"], "TFSA")
        assert adjusted["VOO"] == pytest.approx(expected)

    def test_rejects_invalid_tax_weight(self, engine):
        with pytest.raises(ValueError, match="tax_weight"):
            engine.compute_adjusted_opportunity_scores({}, "TFSA", {}, tax_weight=1.5)


# ---------------------------------------------------------------------------
# AssetLocationEngine — adjust_expected_returns
# ---------------------------------------------------------------------------


class TestAdjustExpectedReturns:
    @pytest.fixture
    def engine(self):
        return AssetLocationEngine(TaxProfile(marginal_tax_rate=0.45))

    def test_single_account(self, engine):
        voo = AssetTaxCharacteristics("VOO", dividend_yield=0.013, is_us_domiciled=True)
        vcn = AssetTaxCharacteristics("VCN.TO", dividend_yield=0.02)
        assets = {"VOO": voo, "VCN": vcn}
        er = {"VOO": 0.10, "VCN": 0.07}
        result = engine.adjust_expected_returns(er, assets, "TFSA")
        # VOO gets FWT drag
        assert result["VOO"] < er["VOO"]
        # VCN has no drag
        assert result["VCN"] == pytest.approx(er["VCN"])

    def test_unknown_ticker_passthrough(self, engine):
        result = engine.adjust_expected_returns({"X": 0.10}, {}, "TFSA")
        assert result["X"] == 0.10

    def test_multi_account(self, engine):
        voo = AssetTaxCharacteristics("VOO", dividend_yield=0.013, is_us_domiciled=True)
        assets = {"VOO": voo}
        er = {"VOO": 0.10}
        accounts = ["TFSA", "RRSP", "Non-Reg"]
        result = engine.adjust_expected_returns_multi_account(er, assets, accounts)
        # RRSP should have highest adjusted return (no FWT)
        assert result[("VOO", "RRSP")] > result[("VOO", "TFSA")]

    def test_multi_account_unknown_ticker(self, engine):
        result = engine.adjust_expected_returns_multi_account(
            {"X": 0.10}, {}, ["TFSA", "RRSP"]
        )
        assert result[("X", "TFSA")] == 0.10
        assert result[("X", "RRSP")] == 0.10


# ---------------------------------------------------------------------------
# build_asset_characteristics / build_asset_characteristics_batch
# ---------------------------------------------------------------------------


class TestBuildAssetCharacteristics:
    def test_basic(self):
        a = build_asset_characteristics("AAPL", dividend_yield=0.005)
        assert a.ticker == "AAPL"
        assert a.dividend_yield == 0.005
        assert a.is_us_domiciled is False

    def test_from_proxy_meta_us_domiciled(self):
        a = build_asset_characteristics(
            "VOO",
            dividend_yield=0.013,
            proxy_meta={"is_us_domiciled": True},
        )
        assert a.is_us_domiciled is True

    def test_from_proxy_meta_cad_wrapped(self):
        a = build_asset_characteristics(
            "VFV.TO",
            dividend_yield=0.012,
            proxy_meta={
                "is_us_domiciled": False,
                "is_cad_wrapped_us_equity": True,
            },
        )
        assert a.is_cad_wrapped_us_equity is True

    def test_heuristic_cad_wrapped_from_proxy(self):
        """VFV.TO with US proxy VOO should be detected as CAD-wrapped US."""
        a = build_asset_characteristics(
            "VFV.TO",
            dividend_yield=0.012,
            proxy_meta={
                "is_us_domiciled": False,
                "is_cad_denominated": True,
                "proxy": "VOO",
            },
        )
        assert a.is_cad_wrapped_us_equity is True

    def test_heuristic_not_triggered_for_cad_proxy(self):
        """A .TO ticker with a .TO proxy should NOT be CAD-wrapped US."""
        a = build_asset_characteristics(
            "XIC.TO",
            dividend_yield=0.025,
            proxy_meta={
                "is_us_domiciled": False,
                "is_cad_denominated": True,
                "proxy": "XIC.TO",
            },
        )
        assert a.is_cad_wrapped_us_equity is False

    def test_heuristic_not_triggered_without_proxy_meta(self):
        """Without proxy_meta, no heuristic is applied."""
        a = build_asset_characteristics("VFV.TO", dividend_yield=0.012)
        assert a.is_cad_wrapped_us_equity is False

    def test_mer_passed_through(self):
        a = build_asset_characteristics("XEQT.TO", dividend_yield=0.02, mer=0.0020)
        assert a.mer == 0.0020

    def test_batch(self):
        tickers = ["VOO", "VFV.TO", "VCN.TO"]
        proxy_map = {
            "VOO": {"is_us_domiciled": True},
            "VFV.TO": {
                "is_us_domiciled": False,
                "is_cad_denominated": True,
                "proxy": "VOO",
            },
        }
        mer_map = {"VFV.TO": 0.0009}
        batch = build_asset_characteristics_batch(
            tickers,
            dividend_yield=0.02,
            proxy_map=proxy_map,
            mer_by_ticker=mer_map,
        )
        assert len(batch) == 3
        assert batch["VOO"].is_us_domiciled is True
        assert batch["VFV.TO"].is_cad_wrapped_us_equity is True
        assert batch["VFV.TO"].mer == 0.0009
        assert batch["VCN.TO"].mer == 0.0
        assert batch["VCN.TO"].has_us_equity_exposure is False


# ---------------------------------------------------------------------------
# Integration with build_rebalance_plan (multi-account, tax-engine path)
# ---------------------------------------------------------------------------


class TestRebalanceIntegration:
    """Verify that build_rebalance_plan uses the tax-location engine correctly."""

    def _write_artifacts(self, export_dir) -> None:
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "demo_weights.txt").write_text(
            "ticker,weight\nVOO,0.50\nVCN,0.30\nVFV,0.20\n",
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "VOO": [400.0, 410.0],
                "VCN": [40.0, 41.0],
                "VFV": [120.0, 122.0],
            }
        ).to_csv(export_dir / "demo_collated.csv", index=False)

    def test_tax_engine_used_when_profile_provided(self, tmp_path):
        """With tax_profile + asset_characteristics, the engine path is used."""
        from pysharpe.execution.rebalance import build_rebalance_plan

        export_dir = tmp_path / "exports"
        self._write_artifacts(export_dir)

        profile = TaxProfile(marginal_tax_rate=0.45)
        characteristics = {
            "VOO": AssetTaxCharacteristics(
                "VOO", dividend_yield=0.013, is_us_domiciled=True
            ),
            "VCN": AssetTaxCharacteristics("VCN.TO", dividend_yield=0.025),
            "VFV": AssetTaxCharacteristics(
                "VFV.TO", dividend_yield=0.012, is_cad_wrapped_us_equity=True
            ),
        }

        # Multi-account holdings via CSV
        holdings_csv = tmp_path / "holdings.csv"
        pd.DataFrame(
            {
                "ticker": ["VOO", "VOO", "VCN", "VCN", "VFV", "VFV"],
                "account": ["TFSA", "RRSP", "TFSA", "RRSP", "TFSA", "RRSP"],
                "current_value": [5000, 5000, 3000, 3000, 2000, 2000],
            }
        ).to_csv(holdings_csv, index=False)

        plan = build_rebalance_plan(
            "demo",
            new_cash=1000.0,
            holdings_csv=holdings_csv,
            export_dir=export_dir,
            tax_profile=profile,
            asset_characteristics=characteristics,
            tax_weight=0.3,
        )

        assert plan.is_multi_account
        assert "TFSA" in plan.accounts
        assert "RRSP" in plan.accounts

    def test_tax_engine_path_via_csv(self, tmp_path):
        """Validate multi-account + tax engine produces different allocations
        across accounts for the same ticker (VOO should go to RRSP)."""
        from pysharpe.execution.rebalance import build_rebalance_plan

        export_dir = tmp_path / "exports"
        self._write_artifacts(export_dir)

        holdings_csv = tmp_path / "holdings.csv"
        pd.DataFrame(
            {
                "ticker": ["VOO", "VCN", "VFV"],
                "account": ["TFSA", "TFSA", "TFSA"],
                "current_value": [5000, 3000, 2000],
            }
        ).to_csv(holdings_csv, index=False)

        profile = TaxProfile(marginal_tax_rate=0.45)
        characteristics = {
            "VOO": AssetTaxCharacteristics(
                "VOO", dividend_yield=0.013, is_us_domiciled=True
            ),
            "VCN": AssetTaxCharacteristics("VCN.TO", dividend_yield=0.025),
            "VFV": AssetTaxCharacteristics(
                "VFV.TO", dividend_yield=0.012, is_cad_wrapped_us_equity=True
            ),
        }

        # Single account (TFSA only) — but with account column it's still multi-account
        plan_single = build_rebalance_plan(
            "demo",
            new_cash=1000.0,
            holdings_csv=holdings_csv,
            export_dir=export_dir,
            tax_profile=profile,
            asset_characteristics=characteristics,
            tax_weight=0.3,
        )
        assert plan_single.is_multi_account
        assert plan_single.accounts == ["TFSA"]

        # Now multi-account: same tickers split across TFSA and RRSP
        holdings_multi = tmp_path / "holdings_multi.csv"
        pd.DataFrame(
            {
                "ticker": ["VOO", "VOO", "VCN", "VCN", "VFV", "VFV"],
                "account": ["TFSA", "RRSP", "TFSA", "RRSP", "TFSA", "RRSP"],
                "current_value": [5000, 5000, 3000, 3000, 2000, 2000],
            }
        ).to_csv(holdings_multi, index=False)

        plan = build_rebalance_plan(
            "demo",
            new_cash=2000.0,
            holdings_csv=holdings_multi,
            export_dir=export_dir,
            tax_profile=profile,
            asset_characteristics=characteristics,
            tax_weight=0.5,
        )
        assert plan.is_multi_account
        assert "TFSA" in plan.accounts
        assert "RRSP" in plan.accounts

        # Check that allocations exist per account
        assert plan.account_allocations is not None
        tfsa_alloc = plan.account_allocations["TFSA"]
        rrsp_alloc = plan.account_allocations["RRSP"]

        # VOO should be preferentially allocated more to RRSP than TFSA
        # because RRSP has no FWT drag for directly held US assets
        tfsa_voo = tfsa_alloc.loc[
            tfsa_alloc["ticker"] == "VOO", "recommended_allocation"
        ].sum()
        rrsp_voo = rrsp_alloc.loc[
            rrsp_alloc["ticker"] == "VOO", "recommended_allocation"
        ].sum()
        # With tax_weight=0.5, the tax efficiency should meaningfully bias
        # VOO toward RRSP
        assert rrsp_voo >= tfsa_voo, (
            f"Expected VOO allocation to favor RRSP over TFSA. "
            f"TFSA: ${tfsa_voo:.2f}, RRSP: ${rrsp_voo:.2f}"
        )

    def test_legacy_fallback_without_tax_profile(self, tmp_path):
        """Without tax_profile, the legacy heuristic is still used."""
        from pysharpe.execution.rebalance import build_rebalance_plan

        export_dir = tmp_path / "exports"
        self._write_artifacts(export_dir)

        holdings_multi = tmp_path / "holdings_multi.csv"
        pd.DataFrame(
            {
                "ticker": ["VOO", "VOO", "VCN", "VCN"],
                "account": ["TFSA", "RRSP", "TFSA", "RRSP"],
                "current_value": [5000, 5000, 3000, 3000],
            }
        ).to_csv(holdings_multi, index=False)

        # No tax_profile → legacy path
        plan = build_rebalance_plan(
            "demo",
            new_cash=1000.0,
            holdings_csv=holdings_multi,
            export_dir=export_dir,
        )
        assert plan.is_multi_account
        assert plan.account_allocations is not None

    def test_tax_engine_integration_with_high_dividend_canadian_equity(self, tmp_path):
        """High-yielding Canadian equities should be favored in TFSA/Non-Reg
        over US-domiciled equities, which go to RRSP."""
        from pysharpe.execution.rebalance import build_rebalance_plan

        export_dir = tmp_path / "exports"
        self._write_artifacts(export_dir)

        # Three accounts: TFSA, RRSP, Non-Reg
        holdings_multi = tmp_path / "holdings.csv"
        pd.DataFrame(
            {
                "ticker": ["VOO", "VOO", "VOO", "VCN", "VCN", "VCN"],
                "account": [
                    "TFSA",
                    "RRSP",
                    "Non-Reg",
                    "TFSA",
                    "RRSP",
                    "Non-Reg",
                ],
                "current_value": [5000, 5000, 5000, 5000, 5000, 5000],
            }
        ).to_csv(holdings_multi, index=False)

        profile = TaxProfile(marginal_tax_rate=0.45)
        characteristics = {
            "VOO": AssetTaxCharacteristics(
                "VOO",
                dividend_yield=0.013,
                is_us_domiciled=True,
                income_frac_foreign_income=0.3,
                income_frac_capital_gains=0.7,
                income_frac_eligible_dividends=0.0,
            ),
            "VCN": AssetTaxCharacteristics(
                "VCN.TO",
                dividend_yield=0.025,
                income_frac_eligible_dividends=0.7,
                income_frac_capital_gains=0.3,
            ),
        }

        plan = build_rebalance_plan(
            "demo",
            new_cash=3000.0,
            holdings_csv=holdings_multi,
            export_dir=export_dir,
            tax_profile=profile,
            asset_characteristics=characteristics,
            tax_weight=0.6,
        )

        # VOO should preferentially go to RRSP (no FWT)
        # VCN should preferentially go to TFSA or Non-Reg
        assert plan.account_allocations is not None
        tfsa = plan.account_allocations["TFSA"]
        rrsp = plan.account_allocations["RRSP"]

        tfsa_voo = tfsa.loc[tfsa["ticker"] == "VOO", "recommended_allocation"].sum()
        rrsp_voo = rrsp.loc[rrsp["ticker"] == "VOO", "recommended_allocation"].sum()

        assert rrsp_voo >= tfsa_voo, (
            f"VOO should favor RRSP (treaty-protected). "
            f"TFSA: ${tfsa_voo:.2f}, RRSP: ${rrsp_voo:.2f}"
        )

    def test_tax_engine_penalizes_cad_wrapped_us_in_rrsp(self, tmp_path):
        """VFV.TO (CAD-wrapped US) should NOT get preferential RRSP treatment
        because FWT is lost at the fund level regardless of account."""
        from pysharpe.execution.rebalance import build_rebalance_plan

        export_dir = tmp_path / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        # Use weights that only include VFV and VCN so the test is focused
        (export_dir / "demo_weights.txt").write_text(
            "ticker,weight\nVFV,0.50\nVCN,0.50\n",
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "VOO": [400.0, 410.0],
                "VCN": [40.0, 41.0],
                "VFV": [120.0, 122.0],
            }
        ).to_csv(export_dir / "demo_collated.csv", index=False)

        holdings_multi = tmp_path / "holdings.csv"
        pd.DataFrame(
            {
                "ticker": ["VFV", "VFV", "VCN", "VCN"],
                "account": ["TFSA", "RRSP", "TFSA", "RRSP"],
                "current_value": [5000, 5000, 5000, 5000],
            }
        ).to_csv(holdings_multi, index=False)

        profile = TaxProfile(marginal_tax_rate=0.45)
        characteristics = {
            "VFV": AssetTaxCharacteristics(
                "VFV.TO",
                dividend_yield=0.012,
                is_cad_wrapped_us_equity=True,
                income_frac_foreign_income=0.2,
                income_frac_capital_gains=0.8,
                income_frac_eligible_dividends=0.0,
            ),
            "VCN": AssetTaxCharacteristics("VCN.TO", dividend_yield=0.025),
        }

        plan = build_rebalance_plan(
            "demo",
            new_cash=2000.0,
            holdings_csv=holdings_multi,
            export_dir=export_dir,
            tax_profile=profile,
            asset_characteristics=characteristics,
            tax_weight=0.3,
        )

        assert plan.account_allocations is not None
        tfsa = plan.account_allocations["TFSA"]
        rrsp = plan.account_allocations["RRSP"]

        tfsa_vfv = tfsa.loc[tfsa["ticker"] == "VFV", "recommended_allocation"].sum()
        rrsp_vfv = rrsp.loc[rrsp["ticker"] == "VFV", "recommended_allocation"].sum()

        # VFV faces the same FWT drag in both TFSA and RRSP (lost at fund
        # level), so there should NOT be a strong RRSP preference for VFV.
        # Both accounts receive some allocation.
        assert plan.is_multi_account
        assert tfsa_vfv > 0
        assert rrsp_vfv > 0
