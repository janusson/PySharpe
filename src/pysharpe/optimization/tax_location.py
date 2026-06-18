"""Asset Location Engine for Canadian tax-advantaged accounts.

Computes account-specific tax drags and tax-adjusted expected returns
for assets placed in TFSA, RRSP, FHSA, and Non-Registered (taxable) accounts.

The engine models three distinct structural Canadian tax rules:

1. **RRSP**: Exempt from US Foreign Withholding Tax (FWT) under the US-Canada
   tax treaty, but only for *directly held* US-domiciled assets (e.g., VOO).
   CAD-wrapped US ETFs (e.g., VFV.TO) lose the 15% FWT at the fund level,
   creating an unrecoverable drag of ``dividend_yield × 0.15``.

2. **TFSA / FHSA**: No treaty protection. Both directly held US-domiciled
   assets and CAD-wrapped US ETFs experience an unrecoverable 15% FWT drag
   on dividend yields.

3. **Non-Registered (Taxable)**: FWT is recoverable via the Foreign Tax Credit,
   but annual income is taxed at marginal rates:
   - Interest / foreign income → taxed at full MTR.
   - Eligible Canadian dividends → gross-up (38%) and dividend tax credit.
   - Capital gains → inclusion rate (50% or 66.67% above $250k threshold).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------


class AccountType(Enum):
    """Canadian registered and non-registered account types."""

    TFSA = "TFSA"
    FHSA = "FHSA"
    RRSP = "RRSP"
    LIRA = "LIRA"
    RRIF = "RRIF"
    NON_REG = "NON_REG"


# Accounts with US-Canada treaty protection against FWT
_TREATY_PROTECTED: frozenset[AccountType] = frozenset(
    {AccountType.RRSP, AccountType.LIRA, AccountType.RRIF}
)

# Registered accounts that lack treaty protection
_NO_TREATY_PROTECTION: frozenset[AccountType] = frozenset(
    {AccountType.TFSA, AccountType.FHSA}
)

# US withholding tax rate on dividends
_US_FWT_RATE: float = 0.15

# Default high-end drag for normalization
_DEFAULT_MAX_DRAG: float = 0.15


def _normalize_account(account: str) -> AccountType:
    """Normalize an account label string to an ``AccountType`` enum member.

    Handles common aliases such as ``"Non-Reg"``, ``"Taxable"``, ``"Margin"``.
    Unknown strings default to ``AccountType.NON_REG``.
    """
    upper = account.upper().strip()
    aliases: dict[str, AccountType] = {
        "NON-REG": AccountType.NON_REG,
        "NON_REG": AccountType.NON_REG,
        "TAXABLE": AccountType.NON_REG,
        "MARGIN": AccountType.NON_REG,
        "CASH": AccountType.NON_REG,
        "UNREGISTERED": AccountType.NON_REG,
    }
    try:
        return AccountType(upper)
    except ValueError:
        return aliases.get(upper, AccountType.NON_REG)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaxProfile:
    """Canadian investor's marginal tax profile.

    Attributes
    ----------
    marginal_tax_rate : float
        Combined federal + provincial marginal tax rate as a decimal
        (e.g. ``0.45`` for 45 %).
    capital_gains_inclusion_rate : float
        Fraction of realised capital gains included in taxable income.
        Standard rate is 0.50 (50 %).  Proposed increase to 0.6667 for
        gains above the $250k annual threshold.
    eligible_dividend_gross_up : float
        Gross-up factor applied to eligible Canadian dividends.  The
        standard federal gross-up is 1.38 (38 %).
    eligible_dividend_tax_credit_federal : float
        Federal dividend tax credit rate applied to the grossed-up amount
        (e.g. ``0.150198`` for the standard 15.0198 % federal DTC).
    eligible_dividend_tax_credit_provincial : float
        Provincial dividend tax credit rate applied to the grossed-up
        amount.  Varies by province; ``0.10`` is a representative value.
    """

    marginal_tax_rate: float
    capital_gains_inclusion_rate: float = 0.50
    eligible_dividend_gross_up: float = 1.38
    eligible_dividend_tax_credit_federal: float = 0.150198
    eligible_dividend_tax_credit_provincial: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 <= self.marginal_tax_rate <= 1.0:
            raise ValueError(
                f"marginal_tax_rate must be in [0, 1], got {self.marginal_tax_rate}"
            )
        if not 0.0 <= self.capital_gains_inclusion_rate <= 1.0:
            raise ValueError(
                "capital_gains_inclusion_rate must be in [0, 1], "
                f"got {self.capital_gains_inclusion_rate}"
            )

    @property
    def capital_gains_effective_rate(self) -> float:
        """Effective tax rate on realised capital gains.

        Computed as ``marginal_tax_rate × capital_gains_inclusion_rate``.
        """
        return self.marginal_tax_rate * self.capital_gains_inclusion_rate

    @property
    def eligible_dividend_effective_rate(self) -> float:
        """Effective tax rate on eligible Canadian dividends.

        Formula: ``MTR × gross_up − (federal_DTC + provincial_DTC)``,
        floored at 0.0.
        """
        grossed_up_tax = self.marginal_tax_rate * self.eligible_dividend_gross_up
        total_dtc = (
            self.eligible_dividend_tax_credit_federal
            + self.eligible_dividend_tax_credit_provincial
        )
        return max(0.0, grossed_up_tax - total_dtc)


@dataclass(frozen=True)
class AssetTaxCharacteristics:
    """Tax-relevant characteristics of an investable asset.

    Attributes
    ----------
    ticker : str
        Asset ticker symbol.
    dividend_yield : float
        Annual dividend yield as a decimal (e.g. ``0.02`` for 2 %).
    is_us_domiciled : bool
        ``True`` when the asset is domiciled in the United States.
    is_cad_wrapped_us_equity : bool
        ``True`` when the asset is a Canadian-listed ETF that wraps US
        equities (e.g. VFV.TO, XUS.TO).  The 15 % US FWT is deducted at
        the fund level and is unrecoverable regardless of account type.
    income_frac_interest : float
        Fraction of expected return attributable to interest / other
        income (taxed at full MTR in non-registered accounts).
    income_frac_eligible_dividends : float
        Fraction of expected return from eligible Canadian dividends.
    income_frac_foreign_income : float
        Fraction of expected return from foreign dividends / income.
    income_frac_capital_gains : float
        Fraction of expected return from realised capital gains.
    mer : float
        Annual Management Expense Ratio as a decimal (e.g. ``0.0017``).

    The four income-fraction fields must sum to 1.0.
    """

    ticker: str
    dividend_yield: float
    is_us_domiciled: bool = False
    is_cad_wrapped_us_equity: bool = False
    income_frac_interest: float = 0.0
    income_frac_eligible_dividends: float = 1.0
    income_frac_foreign_income: float = 0.0
    income_frac_capital_gains: float = 0.0
    mer: float = 0.0

    def __post_init__(self) -> None:
        if self.dividend_yield < 0.0:
            raise ValueError(
                f"dividend_yield must be non-negative, got {self.dividend_yield}"
            )
        total = (
            self.income_frac_interest
            + self.income_frac_eligible_dividends
            + self.income_frac_foreign_income
            + self.income_frac_capital_gains
        )
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Income fractions for {self.ticker!r} must sum to 1.0; got {total}"
            )
        if self.mer < 0.0:
            raise ValueError(f"MER must be non-negative, got {self.mer}")

    @property
    def has_us_equity_exposure(self) -> bool:
        """Return ``True`` when the asset provides exposure to US equities."""
        return self.is_us_domiciled or self.is_cad_wrapped_us_equity


# ---------------------------------------------------------------------------
# AssetLocationEngine
# ---------------------------------------------------------------------------


class AssetLocationEngine:
    """Compute tax-adjusted expected returns across Canadian account types.

    Models the three structural Canadian tax rules governing Foreign
    Withholding Tax and annual income taxation for RRSP, TFSA, FHSA, and
    Non-Registered accounts.

    Parameters
    ----------
    tax_profile : TaxProfile
        The investor's marginal tax rates and dividend-tax-credit parameters.

    Examples
    --------
    >>> profile = TaxProfile(marginal_tax_rate=0.45)
    >>> engine = AssetLocationEngine(profile)
    >>> voo = AssetTaxCharacteristics("VOO", dividend_yield=0.013, is_us_domiciled=True)
    >>> engine.compute_fwt_drag(voo, "RRSP")
    0.0
    >>> engine.compute_fwt_drag(voo, "TFSA")
    0.00195
    >>> engine.compute_total_drag(voo, "TFSA")
    0.00195
    """

    def __init__(self, tax_profile: TaxProfile) -> None:
        self._profile = tax_profile

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_fwt_drag(self, asset: AssetTaxCharacteristics, account: str) -> float:
        """Foreign Withholding Tax drag for *asset* held in *account*.

        Returns the portion of annual dividend yield lost to unrecoverable
        US withholding tax.

        Parameters
        ----------
        asset : AssetTaxCharacteristics
        account : str
            Account label (e.g. ``"TFSA"``, ``"RRSP"``, ``"Non-Reg"``).

        Returns
        -------
        float
            Annual FWT drag as a decimal (e.g. ``0.00195`` for 0.195 %).
        """
        if not asset.has_us_equity_exposure:
            return 0.0

        acct = _normalize_account(account)

        if acct in _TREATY_PROTECTED:
            # RRSP / LIRA / RRIF: US-Canada treaty exempts directly-held
            # US-domiciled assets from FWT.
            if asset.is_us_domiciled and not asset.is_cad_wrapped_us_equity:
                return 0.0
            # CAD-wrapped US equity: FWT is taken at the fund level and is
            # unrecoverable in *any* account type.
            if asset.is_cad_wrapped_us_equity:
                return asset.dividend_yield * _US_FWT_RATE
            return 0.0

        if acct in _NO_TREATY_PROTECTION:
            # TFSA / FHSA: no treaty → FWT is lost on all US equity exposure.
            return asset.dividend_yield * _US_FWT_RATE

        # Non-Registered: FWT is recoverable via the Foreign Tax Credit.
        return 0.0

    def compute_income_tax_drag(
        self, asset: AssetTaxCharacteristics, account: str
    ) -> float:
        """Annual income-tax drag for *asset* held in *account*.

        Only applies to Non-Registered (taxable) accounts.  Registered
        accounts (TFSA, RRSP, FHSA) defer or eliminate annual taxation,
        so their income-tax drag is zero.

        Parameters
        ----------
        asset : AssetTaxCharacteristics
        account : str

        Returns
        -------
        float
            Annual income-tax drag as a decimal fraction of expected return.
        """
        acct = _normalize_account(account)
        if acct != AccountType.NON_REG:
            return 0.0

        tp = self._profile
        drag = 0.0

        drag += asset.income_frac_interest * tp.marginal_tax_rate
        drag += (
            asset.income_frac_eligible_dividends * tp.eligible_dividend_effective_rate
        )
        drag += asset.income_frac_foreign_income * tp.marginal_tax_rate
        drag += asset.income_frac_capital_gains * tp.capital_gains_effective_rate

        return drag

    def compute_total_drag(self, asset: AssetTaxCharacteristics, account: str) -> float:
        """Sum of FWT and income-tax drags for *asset* in *account*.

        This is :math:`\\text{Drag}_{i,A}` in the tax-adjustment formula:

        .. math::

            R_{i,A} = R_i - \\text{MER}_i - \\text{Drag}_{i,A}
        """
        return self.compute_fwt_drag(asset, account) + self.compute_income_tax_drag(
            asset, account
        )

    def compute_tax_adjusted_return(
        self,
        expected_return: float,
        asset: AssetTaxCharacteristics,
        account: str,
    ) -> float:
        """Tax-adjusted annualized expected return.

        .. math::

            R_{i,A} = R_i - \\text{MER}_i - \\text{Drag}_{i,A}

        Parameters
        ----------
        expected_return : float
            Pre-tax annualized expected return as a decimal.
        asset : AssetTaxCharacteristics
        account : str

        Returns
        -------
        float
            Tax- and MER-adjusted annualized expected return.
        """
        drag = self.compute_total_drag(asset, account)
        return expected_return - asset.mer - drag

    def compute_tax_efficiency_score(
        self,
        asset: AssetTaxCharacteristics,
        account: str,
        *,
        max_drag: float = _DEFAULT_MAX_DRAG,
    ) -> float:
        """Normalised [0, 1] tax-efficiency score.

        A score of ``1.0`` indicates the asset is maximally tax-efficient
        in the given account (lowest drag).  ``0.0`` corresponds to the
        worst-case drag within the normalisation window.

        Parameters
        ----------
        asset : AssetTaxCharacteristics
        account : str
        max_drag : float
            Upper bound for drag normalisation.  Drags above this value
            are clamped.

        Returns
        -------
        float
            Tax efficiency score in **[0, 1]**.
        """
        if max_drag <= 0.0:
            raise ValueError(f"max_drag must be positive, got {max_drag}")
        drag = self.compute_total_drag(asset, account)
        clamped = min(drag, max_drag)
        return 1.0 - (clamped / max_drag)

    def compute_adjusted_opportunity_scores(
        self,
        assets: dict[str, AssetTaxCharacteristics],
        account: str,
        base_scores: dict[str, float],
        *,
        tax_weight: float = 0.3,
    ) -> dict[str, float]:
        """Blend pre-tax opportunity scores with tax efficiency.

        For each ticker in *base_scores* the adjusted score is:

        .. math::

            (1 - w_t) \\times \\text{base} + w_t \\times \\text{tax\\_efficiency}

        where :math:`w_t` is *tax_weight*.

        Parameters
        ----------
        assets : dict[str, AssetTaxCharacteristics]
            Ticker → asset characteristics lookup.
        account : str
        base_scores : dict[str, float]
            Pre-tax opportunity scores keyed by ticker.
        tax_weight : float
            Weight assigned to tax efficiency in the blended score (0–1).

        Returns
        -------
        dict[str, float]
        """
        if not 0.0 <= tax_weight <= 1.0:
            raise ValueError(f"tax_weight must be in [0, 1], got {tax_weight}")

        adjusted: dict[str, float] = {}
        for ticker, base in base_scores.items():
            asset = assets.get(ticker)
            if asset is None:
                adjusted[ticker] = base
                continue
            tax_eff = self.compute_tax_efficiency_score(asset, account)
            adjusted[ticker] = (1.0 - tax_weight) * base + tax_weight * tax_eff
        return adjusted

    def adjust_expected_returns(
        self,
        expected_returns: dict[str, float],
        assets: dict[str, AssetTaxCharacteristics],
        account: str,
    ) -> dict[str, float]:
        """Compute tax-adjusted expected returns for all assets in an account.

        Parameters
        ----------
        expected_returns : dict[str, float]
            Ticker → annualized expected return.
        assets : dict[str, AssetTaxCharacteristics]
        account : str

        Returns
        -------
        dict[str, float]
            Ticker → tax-adjusted expected return.
        """
        return {
            ticker: self.compute_tax_adjusted_return(r, assets[ticker], account)
            if ticker in assets
            else r
            for ticker, r in expected_returns.items()
        }

    def adjust_expected_returns_multi_account(
        self,
        expected_returns: dict[str, float],
        assets: dict[str, AssetTaxCharacteristics],
        accounts: list[str],
    ) -> dict[tuple[str, str], float]:
        """Compute tax-adjusted returns for all (ticker, account) pairs.

        Parameters
        ----------
        expected_returns : dict[str, float]
        assets : dict[str, AssetTaxCharacteristics]
        accounts : list[str]
            Account labels to evaluate.

        Returns
        -------
        dict[tuple[str, str], float]
            Mapping from ``(ticker, account)`` to adjusted return.
        """
        result: dict[tuple[str, str], float] = {}
        for ticker, r in expected_returns.items():
            asset = assets.get(ticker)
            if asset is None:
                for acct in accounts:
                    result[(ticker, acct)] = r
                continue
            for acct in accounts:
                result[(ticker, acct)] = self.compute_tax_adjusted_return(
                    r, asset, acct
                )
        return result


# ---------------------------------------------------------------------------
# Helper: build AssetTaxCharacteristics from proxy_map metadata
# ---------------------------------------------------------------------------


def build_asset_characteristics(
    ticker: str,
    *,
    dividend_yield: float = 0.02,
    proxy_meta: dict[str, object] | None = None,
    mer: float = 0.0,
    income_frac_interest: float = 0.0,
    income_frac_eligible_dividends: float = 1.0,
    income_frac_foreign_income: float = 0.0,
    income_frac_capital_gains: float = 0.0,
) -> AssetTaxCharacteristics:
    """Build an ``AssetTaxCharacteristics`` from explicit values and proxy metadata.

    When *proxy_meta* is provided the domicile and wrapper flags are read
    from it; otherwise sensible defaults are assumed.

    Parameters
    ----------
    ticker : str
    dividend_yield : float
        Annual dividend yield estimate (decimal).
    proxy_meta : dict or None
        A single entry from the project's ``proxy_map.json``.
    mer : float
        Annual MER (decimal).
    income_frac_interest, income_frac_eligible_dividends,
    income_frac_foreign_income, income_frac_capital_gains : float
        Income-composition fractions (must sum to 1.0).

    Returns
    -------
    AssetTaxCharacteristics
    """
    is_us = False
    is_cad_wrapped = False

    if proxy_meta is not None:
        is_us = bool(proxy_meta.get("is_us_domiciled", False))
        is_cad_wrapped = bool(proxy_meta.get("is_cad_wrapped_us_equity", False))

    # Heuristic: .TO ticker with a US proxy → CAD-wrapped US equity
    if not is_us and not is_cad_wrapped:
        if ticker.upper().endswith(".TO") and proxy_meta is not None:
            proxy = proxy_meta.get("proxy")
            if proxy is not None and isinstance(proxy, str):
                # If the proxy ticker doesn't end in .TO, it's likely US
                if not proxy.upper().endswith(".TO"):
                    is_cad_wrapped = True

    return AssetTaxCharacteristics(
        ticker=ticker,
        dividend_yield=dividend_yield,
        is_us_domiciled=is_us,
        is_cad_wrapped_us_equity=is_cad_wrapped,
        income_frac_interest=income_frac_interest,
        income_frac_eligible_dividends=income_frac_eligible_dividends,
        income_frac_foreign_income=income_frac_foreign_income,
        income_frac_capital_gains=income_frac_capital_gains,
        mer=mer,
    )


def build_asset_characteristics_batch(
    tickers: list[str],
    *,
    dividend_yield: float = 0.02,
    proxy_map: dict[str, dict[str, object]] | None = None,
    mer_by_ticker: dict[str, float] | None = None,
    **income_kwargs: float,
) -> dict[str, AssetTaxCharacteristics]:
    """Build a batch of ``AssetTaxCharacteristics`` from a ticker list.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to build characteristics for.
    dividend_yield : float
        Default annual dividend yield for tickers without a custom value.
    proxy_map : dict or None
        Full proxy map; each entry's metadata is used to infer domicile
        and wrapper status.
    mer_by_ticker : dict or None
        Per-ticker MER values.
    **income_kwargs
        Forwarded to ``build_asset_characteristics`` as income-fraction
        defaults.

    Returns
    -------
    dict[str, AssetTaxCharacteristics]
    """
    proxy_map = proxy_map or {}
    mer_by_ticker = mer_by_ticker or {}

    result: dict[str, AssetTaxCharacteristics] = {}
    for ticker in tickers:
        proxy_meta = proxy_map.get(ticker)
        mer = mer_by_ticker.get(ticker, 0.0)
        result[ticker] = build_asset_characteristics(
            ticker,
            dividend_yield=dividend_yield,
            proxy_meta=proxy_meta,
            mer=mer,
            **income_kwargs,
        )
    return result
