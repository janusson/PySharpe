"""Portfolio X-ray: what a tracker cannot tell you about the portfolio you own.

A record-keeping system reports what happened. The X-ray reads a
:class:`~pysharpe.data.interop.HoldingsSnapshot` and reports what the current
shape of the portfolio *implies*:

- **Identical exposure** — the same underlying bet held more than once, across
  accounts (``VFV.TO`` and ``VOO`` are the same index), using the
  identical-property map that also drives superficial-loss compliance.
- **Concentration** — largest position weight and effective number of
  holdings.
- **Currency exposure** — CAD versus foreign exposure in CAD, at the snapshot's
  own FX rates.
- **Tax location** — foreign-withholding drag on assets held in TFSA/FHSA and
  annual income tax drag in non-registered accounts, in basis points, with the
  account that would avoid the drag when the snapshot contains one.

Everything is computed from the snapshot alone. Flags are annualized drag
estimates in basis points of the position's yield, not tax advice, and unknown
inputs stay unknown: a ticker without
:class:`~pysharpe.optimization.tax_location.AssetTaxCharacteristics` is
reported as unassessed rather than assumed tax-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

from pysharpe.data.interop import HoldingsSnapshot
from pysharpe.exceptions import DataValidationError
from pysharpe.guardrails.tax_compliance import build_default_identical_map
from pysharpe.optimization.tax_location import (
    AccountType,
    AssetLocationEngine,
    AssetTaxCharacteristics,
    TaxProfile,
)

#: Assumed marginal rate when the caller supplies none. Reported in the output.
_DEFAULT_MARGINAL_RATE = 0.40

#: Default share of securities value above which a position is called out.
_DEFAULT_CONCENTRATION_THRESHOLD = 0.25

_NON_REG_ALIASES: dict[str, AccountType] = {
    "NON-REG": AccountType.NON_REG,
    "NONREG": AccountType.NON_REG,
    "TAXABLE": AccountType.NON_REG,
    "MARGIN": AccountType.NON_REG,
    "UNREGISTERED": AccountType.NON_REG,
    "CASH": AccountType.NON_REG,
}

#: Accounts with US treaty protection against foreign withholding tax.
_TREATY_ACCOUNTS = frozenset({AccountType.RRSP, AccountType.LIRA, AccountType.RRIF})


def _resolve_account_type(label: str) -> AccountType:
    """Map an account label to an :class:`AccountType` without silent defaulting.

    The tax-location engine defaults unknown labels to non-registered, which
    would quietly mis-state a TFSA as taxable. The X-ray refuses instead.
    """
    text = label.strip().upper()
    try:
        return AccountType(text)
    except ValueError:
        pass
    if text in _NON_REG_ALIASES:
        return _NON_REG_ALIASES[text]
    allowed = ", ".join(member.value for member in AccountType)
    raise DataValidationError(
        f"Unknown account type {label!r}; expected one of {allowed} "
        f"(aliases: {', '.join(sorted(_NON_REG_ALIASES))})."
    )


def _identical_groups(
    identical_map: Mapping[str, frozenset[str]],
) -> dict[str, frozenset[str]]:
    """Return ticker -> its identical-property group, as a symmetric closure."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for ticker, peers in identical_map.items():
        find(ticker)
        for peer in peers:
            union(ticker, peer)

    groups: dict[str, set[str]] = {}
    for ticker in parent:
        groups.setdefault(find(ticker), set()).add(ticker)
    return {
        ticker: frozenset(members) for members in groups.values() for ticker in members
    }


@dataclass(frozen=True)
class XrayFlag:
    """One observation about the current portfolio.

    Attributes
    ----------
    code
        Stable identifier: ``"identical_exposure"``, ``"concentration"``,
        ``"withholding_drag"``, ``"income_tax_drag"``, ``"unassessed_asset"``.
    severity
        ``"warning"`` for actionable exposure, ``"info"`` otherwise.
    message
        Human-readable explanation, including the numbers that triggered it.
    tickers
        Tickers involved.
    accounts
        Account labels involved.
    drag_bps
        Annualized drag in basis points when the flag quantifies one.
    """

    code: str
    severity: str
    message: str
    tickers: tuple[str, ...] = ()
    accounts: tuple[str, ...] = ()
    drag_bps: float | None = None


@dataclass(frozen=True)
class XrayReport:
    """Result of :func:`portfolio_xray`.

    Attributes
    ----------
    as_of
        Date of the snapshot the report describes.
    total_value_cad
        Securities plus cash, in CAD.
    cash_value_cad
        Cash component, in CAD.
    positions
        Per-position table: ``account``, ``ticker``, ``currency``, ``quantity``,
        ``value_cad``, ``weight`` (share of securities value).
    currency_exposure
        Per-currency table: ``currency``, ``value_cad``, ``weight`` (share of
        total value, cash included).
    identical_groups
        Groups of ``(account, ticker)`` pairs that hold identical property.
    flags
        Observations, warnings first.
    largest_position_weight
        Largest share of securities value held in one position.
    effective_holdings
        ``1 / sum(weight**2)`` over securities; 0.0 when no securities.
    marginal_tax_rate
        Marginal rate used for income-tax drag estimates.
    """

    as_of: date
    total_value_cad: float
    cash_value_cad: float
    positions: pd.DataFrame
    currency_exposure: pd.DataFrame
    identical_groups: tuple[tuple[tuple[str, str], ...], ...]
    flags: tuple[XrayFlag, ...]
    largest_position_weight: float
    effective_holdings: float
    marginal_tax_rate: float

    def to_frame(self) -> pd.DataFrame:
        """Return the flags as a DataFrame, warnings first."""
        return pd.DataFrame(
            [
                {
                    "code": flag.code,
                    "severity": flag.severity,
                    "tickers": ", ".join(flag.tickers),
                    "accounts": ", ".join(flag.accounts),
                    "drag_bps": flag.drag_bps,
                    "message": flag.message,
                }
                for flag in self.flags
            ],
            columns=[
                "code",
                "severity",
                "tickers",
                "accounts",
                "drag_bps",
                "message",
            ],
        )

    def report(self) -> str:
        """Return a compact text report."""
        lines = [
            f"PORTFOLIO X-RAY (as of {self.as_of.isoformat()})",
            "=" * 62,
            f"total value          {self.total_value_cad:>12,.2f} CAD",
            f"  of which cash      {self.cash_value_cad:>12,.2f} CAD",
            f"largest position     {self.largest_position_weight:>11.2%}",
            f"effective holdings   {self.effective_holdings:>11.1f}",
            f"marginal rate used   {self.marginal_tax_rate:>11.1%}",
            "",
            "CURRENCY EXPOSURE",
        ]
        for row in self.currency_exposure.itertuples(index=False):
            lines.append(
                f"  {row.currency:<6}{row.value_cad:>12,.2f} CAD{row.weight:>9.2%}"
            )

        lines.append("")
        lines.append("POSITIONS")
        for row in self.positions.itertuples(index=False):
            lines.append(
                f"  {row.account:<12}{row.ticker:<12}{row.value_cad:>12,.2f} CAD"
                f"{row.weight:>9.2%}"
            )

        if self.identical_groups:
            lines.append("")
            lines.append("IDENTICAL EXPOSURE")
            for group in self.identical_groups:
                members = ", ".join(
                    f"{ticker} ({account})" for account, ticker in group
                )
                lines.append(f"  {members}")

        lines.append("")
        if self.flags:
            lines.append(f"FINDINGS ({len(self.flags)})")
            for flag in self.flags:
                drag = "" if flag.drag_bps is None else f" [{flag.drag_bps:,.1f} bps]"
                lines.append(f"  [{flag.severity}] {flag.code}{drag}")
                lines.append(f"    {flag.message}")
        else:
            lines.append("FINDINGS: nothing to report for this snapshot")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _position_table(
    snapshot: HoldingsSnapshot,
) -> tuple[pd.DataFrame, float]:
    """Return the per-position table and total securities value."""
    rows = [
        {
            "account": position.account,
            "ticker": position.ticker,
            "currency": position.currency,
            "quantity": position.quantity,
            "value_cad": snapshot.position_value_cad(position),
        }
        for position in snapshot.positions
    ]
    table = pd.DataFrame(
        rows,
        columns=["account", "ticker", "currency", "quantity", "value_cad"],
    )
    securities_value = float(table["value_cad"].sum()) if not table.empty else 0.0
    if securities_value > 0.0:
        table["weight"] = table["value_cad"] / securities_value
    else:
        table["weight"] = 0.0
    table = table.sort_values(
        ["value_cad", "account", "ticker"],
        ascending=[False, True, True],
        ignore_index=True,
    )
    return table, securities_value


def _currency_table(
    snapshot: HoldingsSnapshot,
    total_value_cad: float,
) -> pd.DataFrame:
    """Return per-currency CAD exposure, securities plus cash."""
    values: dict[str, float] = {}
    for position in snapshot.positions:
        values[position.currency] = values.get(position.currency, 0.0) + (
            snapshot.position_value_cad(position)
        )
    for balances in snapshot.cash.values():
        for currency, amount in balances.items():
            code = currency.strip().upper()
            values[code] = values.get(code, 0.0) + amount * snapshot.rate(code)

    rows = [
        {
            "currency": code,
            "value_cad": value,
            "weight": (value / total_value_cad) if total_value_cad > 0.0 else 0.0,
        }
        for code, value in sorted(values.items())
    ]
    return pd.DataFrame(rows, columns=["currency", "value_cad", "weight"])


def _group_members(
    table: pd.DataFrame,
    groups: Mapping[str, frozenset[str]],
) -> list[list[tuple[str, str]]]:
    """Group ``(account, ticker)`` pairs by identical-property compound."""
    buckets: dict[frozenset[str], list[tuple[str, str]]] = {}
    for row in table.itertuples(index=False):
        key = groups.get(str(row.ticker), frozenset({str(row.ticker)}))
        buckets.setdefault(key, []).append((str(row.account), str(row.ticker)))

    result: list[list[tuple[str, str]]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        tickers = {ticker for _, ticker in members}
        accounts = {account for account, _ in members}
        if len(tickers) < 2 and len(accounts) < 2:
            continue
        result.append(sorted(members))
    return sorted(result)


def _tax_flags(
    table: pd.DataFrame,
    account_types: Mapping[str, AccountType],
    characteristics: Mapping[str, AssetTaxCharacteristics],
    engine: AssetLocationEngine,
) -> list[XrayFlag]:
    """Build withholding and income-tax drag flags from the snapshot."""
    flags: list[XrayFlag] = []
    treaty_accounts = sorted(
        name
        for name, account_type in account_types.items()
        if account_type in _TREATY_ACCOUNTS
    )

    for row in table.itertuples(index=False):
        account_type = account_types[str(row.account)]
        asset = characteristics.get(str(row.ticker))
        if asset is None:
            continue

        fwt = engine.compute_fwt_drag(asset, account_type.value)
        if fwt > 0.0:
            message = (
                f"{row.ticker} in {row.account} loses {fwt:.4%} of yield per year "
                f"to US foreign withholding tax."
            )
            if asset.is_cad_wrapped_us_equity:
                message += (
                    " The tax is withheld inside the fund, so no account "
                    "wrapper recovers it."
                )
            else:
                message += " TFSA and FHSA have no treaty protection."
                if treaty_accounts:
                    message += (
                        " Directly held US-domiciled assets avoid this in "
                        + ", ".join(treaty_accounts)
                        + "."
                    )
            flags.append(
                XrayFlag(
                    code="withholding_drag",
                    severity="warning",
                    message=message,
                    tickers=(str(row.ticker),),
                    accounts=(str(row.account),),
                    drag_bps=round(fwt * 10_000.0, 2),
                )
            )

        if account_type is AccountType.NON_REG:
            income_drag = engine.compute_income_tax_drag(asset, account_type.value)
            if income_drag <= 0.0:
                continue
            flags.append(
                XrayFlag(
                    code="income_tax_drag",
                    severity="info",
                    message=(
                        f"{row.ticker} in {row.account} is taxed on income; the "
                        f"annual income-tax drag is {income_drag:.4%} of the "
                        f"income component and is not recoverable."
                    ),
                    tickers=(str(row.ticker),),
                    accounts=(str(row.account),),
                    drag_bps=round(income_drag * 10_000.0, 2),
                )
            )
    return flags


def portfolio_xray(
    snapshot: HoldingsSnapshot,
    *,
    account_types: Mapping[str, str],
    asset_characteristics: Mapping[str, AssetTaxCharacteristics] | None = None,
    tax_profile: TaxProfile | None = None,
    identical_map: Mapping[str, frozenset[str]] | None = None,
    concentration_threshold: float = _DEFAULT_CONCENTRATION_THRESHOLD,
) -> XrayReport:
    """Report the shape, concentration, and tax placement of a holdings snapshot.

    Args:
        snapshot: Imported holdings snapshot.
        account_types: Account label -> account type (``"TFSA"``, ``"RRSP"``,
            ``"NON_REG"``, ...). Every account in the snapshot, including cash
            accounts, must be listed; unknown labels raise rather than default
            to non-registered.
        asset_characteristics: Ticker -> tax characteristics used for drag
            estimates. Tickers without an entry are reported as unassessed.
        tax_profile: Marginal tax profile used for income-tax drag. Defaults to
            a 40% marginal rate, which the report states explicitly.
        identical_map: Ticker -> identical-property peers. Defaults to
            :func:`~pysharpe.guardrails.tax_compliance.build_default_identical_map`.
        concentration_threshold: Share of securities value above which the
            largest position is flagged.

    Returns:
        XrayReport with per-position, per-currency, and flag tables.

    Raises:
        DataValidationError: If an account is missing from ``account_types`` or
            carries an unknown type, or if the threshold is out of range.
    """
    if not 0.0 < concentration_threshold <= 1.0:
        raise DataValidationError(
            f"concentration_threshold must be in (0, 1]; got {concentration_threshold}."
        )

    snapshot_accounts = {position.account for position in snapshot.positions} | set(
        snapshot.cash
    )
    missing = sorted(snapshot_accounts - set(account_types))
    if missing:
        raise DataValidationError(
            "account_types is missing accounts present in the snapshot: "
            + ", ".join(missing)
            + ". Declare each account's type explicitly; PySharpe will not "
            "assume an account is taxable."
        )
    resolved_types = {
        account: _resolve_account_type(label)
        for account, label in account_types.items()
    }

    table, securities_value = _position_table(snapshot)
    cash_value = snapshot.cash_value_cad()
    total_value = snapshot.total_value_cad()

    weights = [float(weight) for weight in table["weight"].tolist()]
    largest = max(weights) if weights else 0.0
    hhi = sum(weight * weight for weight in weights)
    effective_holdings = (1.0 / hhi) if hhi > 0.0 else 0.0

    closure = _identical_groups(identical_map or build_default_identical_map())
    grouped = _group_members(table, closure)

    flags: list[XrayFlag] = []
    for group in grouped:
        tickers = tuple(sorted({ticker for _, ticker in group}))
        accounts = tuple(sorted({account for account, _ in group}))
        detail = ", ".join(f"{ticker} ({account})" for account, ticker in group)
        same_asset = len(tickers) == 1
        flags.append(
            XrayFlag(
                code="identical_exposure",
                severity="warning",
                message=(
                    f"Identical property held in {len(group)} places: {detail}. "
                    + (
                        "The same security sits in multiple accounts, so "
                        "superficial-loss rules span all of them."
                        if same_asset
                        else "Different tickers, same underlying exposure: "
                        "diversification is overstated and gains are "
                        "correlated."
                    )
                ),
                tickers=tickers,
                accounts=accounts,
            )
        )

    if securities_value > 0.0 and largest > concentration_threshold:
        top_ticker = str(table.loc[0, "ticker"])
        top_account = str(table.loc[0, "account"])
        flags.append(
            XrayFlag(
                code="concentration",
                severity="warning",
                message=(
                    f"{top_ticker} in {top_account} is {largest:.2%} of "
                    f"securities value, above the {concentration_threshold:.0%} "
                    f"threshold; effective number of holdings is "
                    f"{effective_holdings:.1f}."
                ),
                tickers=(top_ticker,),
                accounts=(top_account,),
            )
        )

    characteristics = asset_characteristics or {}
    profile = tax_profile or TaxProfile(marginal_tax_rate=_DEFAULT_MARGINAL_RATE)
    engine = AssetLocationEngine(profile)
    flags.extend(_tax_flags(table, resolved_types, characteristics, engine))

    unassessed = sorted(
        ticker
        for ticker in table["ticker"].unique()
        if str(ticker) not in characteristics
    )
    if unassessed:
        flags.append(
            XrayFlag(
                code="unassessed_asset",
                severity="info",
                message=(
                    "No tax characteristics supplied for "
                    + ", ".join(unassessed)
                    + "; their withholding and income drag are unknown, not "
                    "zero. Pass asset_characteristics to assess them."
                ),
                tickers=tuple(unassessed),
            )
        )

    severity_rank = {"warning": 0, "info": 1}
    flags.sort(
        key=lambda flag: (severity_rank.get(flag.severity, 2), flag.code, flag.tickers)
    )

    return XrayReport(
        as_of=snapshot.as_of,
        total_value_cad=total_value,
        cash_value_cad=cash_value,
        positions=table,
        currency_exposure=_currency_table(snapshot, total_value),
        identical_groups=tuple(tuple(group) for group in grouped),
        flags=tuple(flags),
        largest_position_weight=largest,
        effective_holdings=effective_holdings,
        marginal_tax_rate=profile.marginal_tax_rate,
    )


__all__ = ["XrayFlag", "XrayReport", "portfolio_xray"]
