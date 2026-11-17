"""Vendor-neutral holdings snapshot import.

PySharpe is **not** a book of record. This module imports a point-in-time
holdings snapshot from any export that can be mapped onto the canonical column
names below, and adapts it to the schemas the allocation and rebalance engines
already consume. Transaction histories, dividends, corporate actions, and tax
reporting stay with the record-keeping system that produced the file.

Canonical snapshot columns
--------------------------
==================  ======  ====================================================
canonical name      type    meaning
==================  ======  ====================================================
``ticker``          str     instrument identifier as exported
``account``         str     account label from the exporting system
``quantity``        float   units held; must be strictly positive
``price``           float   price per unit in ``currency`` at the snapshot date
``currency``        str     ISO 4217 code of ``price``; ``"CAD"`` is the default
``cost_basis``      float   optional CAD cost basis; blank means *unknown*
==================  ======  ====================================================

Rules
-----
- **FX coverage is explicit.** Every non-CAD currency in the snapshot needs a
  rate in ``fx_rates`` (CAD per unit), and a snapshot with an uncovered
  currency is rejected at construction/import time. Nothing is backfilled,
  consistent with the package-wide no-``bfill()`` rule.
- **Failures are loud.** Duplicate ``(account, ticker)`` rows, non-positive
  quantities, non-positive prices, and unparseable numbers raise
  :class:`~pysharpe.exceptions.DataValidationError` naming the offending row.
- **A blank cost basis stays unknown.** It is never coerced to ``0.0``;
  :meth:`HoldingsSnapshot.missing_cost_basis` lists what is unknown.
- **Structural rows are skipped, not guessed.** Report exports carry group
  headers and subtotal/footer rows. A row with neither a quantity nor a price
  cannot be a position, so it is skipped; everything else must parse.
- **No network access.** Prices are exactly what the export says; PySharpe does
  not refresh them, and ``as_of`` is caller-supplied rather than inferred.

Downstream contracts
--------------------
:meth:`HoldingsSnapshot.current_balances` matches the ``current_balances``
argument of
:func:`pysharpe.execution.cash_flow_rebalance.allocate_contribution_cash_flow`
(columns ``ticker``, ``account``, ``current_value``).
:meth:`HoldingsSnapshot.holdings_frame` matches the ``--holdings-csv``
convention accepted by :func:`pysharpe.execution.rebalance.build_rebalance_plan`
(``ticker`` plus ``shares``/``current_value``, optional ``account``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from pysharpe.exceptions import DataIngestionError, DataValidationError

logger = logging.getLogger(__name__)

#: Canonical column names a snapshot mapping must supply.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "ticker",
    "account",
    "quantity",
    "price",
    "currency",
)

#: Canonical column names that may be supplied but need not be.
OPTIONAL_COLUMNS: tuple[str, ...] = ("cost_basis",)

#: Header labels recognised by :func:`infer_column_mapping`, per canonical name.
#: Matching ignores case, punctuation, and parenthetical qualifiers, so
#: ``"Cost Basis (CA$)"`` matches ``"cost basis"``.
_CANONICAL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "ticker": ("ticker", "symbol", "instrument", "instrument code", "security"),
    "account": ("account", "holding account", "account name", "portfolio account"),
    "quantity": ("quantity", "units", "units held", "quantity held", "shares"),
    "price": ("price", "current price", "price per unit", "market price"),
    "currency": ("currency", "currency code", "ccy"),
    "cost_basis": ("cost basis", "cost base", "book cost", "acb", "total cost"),
}

_CAD = "CAD"

#: Market codes seen in portfolio exports, mapped to the yfinance-style suffix
#: PySharpe prices with. ``""`` means the US listing needs no suffix. Codes
#: outside this map are left untouched, so an unknown market never silently
#: rewrites an identifier.
MARKET_SUFFIXES: dict[str, str] = {
    "TSE": ".TO",
    "TSX": ".TO",
    "CVE": ".V",
    "TSXV": ".V",
    "NYSE": "",
    "NASDAQ": "",
    "NYSEARCA": "",
    "ARCA": "",
    "AMEX": "",
    "BATS": "",
}


def normalise_ticker(
    ticker: str,
    *,
    markets: Mapping[str, str] | None = None,
) -> str:
    """Convert an exported instrument label to the ticker PySharpe prices with.

    Portfolio exports append a *market code* to the instrument (``VFV.TSE``,
    ``VOO.NYSE``, or ``VFV | TSE`` in dashboard labels), while prices are keyed
    by the yfinance convention (``VFV.TO``, ``VOO``). This rewrites the market
    code only when it is recognised:

    >>> normalise_ticker("VFV.TSE")
    'VFV.TO'
    >>> normalise_ticker("VOO.NYSE")
    'VOO'
    >>> normalise_ticker("VFV | TSE")
    'VFV.TO'

    Identifiers with no market code, or with an unrecognised one (a share class
    such as ``BRK.B``, or an already-normalised ``VFV.TO``) are returned
    unchanged. PySharpe never guesses an exchange.

    Args:
        ticker: Instrument label as exported.
        markets: Market code -> yfinance suffix. Defaults to
            :data:`MARKET_SUFFIXES`; pass a mapping to extend it for other
            exchanges.

    Returns:
        The normalised ticker, or the input unchanged when it already looks
        final.

    Raises:
        DataValidationError: If *ticker* is empty, or if stripping the market
            code leaves no instrument.
    """
    text = str(ticker).strip()
    if not text:
        raise DataValidationError(
            "Ticker is empty; every imported position needs an identifier."
        )

    if "|" in text:
        base, _, market = text.rpartition("|")
    elif "." in text:
        base, _, market = text.rpartition(".")
    else:
        return text

    base = base.strip()
    market = market.strip().upper()
    table = markets if markets is not None else MARKET_SUFFIXES
    suffix = table.get(market)
    if suffix is None:
        return text
    if not base:
        raise DataValidationError(
            f"Ticker {ticker!r} has a market code but no instrument name."
        )
    if suffix and base.upper().endswith(suffix.upper()):
        return base
    return f"{base}{suffix}"


def _normalise_header(header: object) -> str:
    """Reduce a header label to lowercase words for synonym matching."""
    text = re.sub(r"\([^)]*\)", " ", str(header).strip().lower())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def infer_column_mapping(
    columns: Sequence[object],
    *,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Guess the ``columns`` mapping for an export from its header row.

    Matching is by documented synonym, so header drift between vendors (and
    between exports of the same vendor) does not need a code change. Ambiguous
    matches are an error rather than a coin flip.

    Parameters
    ----------
    columns
        Header labels as read from the file.
    overrides
        Canonical name -> source header; wins over inference.

    Returns
    -------
    dict[str, str]
        The subset of canonical names that matched, plus any overrides. The
        caller still has to satisfy ``REQUIRED_COLUMNS``.

    Raises
    ------
    DataValidationError
        If two columns match the same canonical name.

    Example:
        >>> infer_column_mapping(["Instrument", "Units Held", "Current Price"])
        {'ticker': 'Instrument', 'quantity': 'Units Held', 'price': 'Current Price'}
    """
    available = {_normalise_header(column): column for column in columns}
    mapping: dict[str, str] = {}

    for canonical, synonyms in _CANONICAL_SYNONYMS.items():
        if overrides and canonical in overrides:
            continue  # An explicit override also resolves ambiguity.
        matches = [str(available[alias]) for alias in synonyms if alias in available]
        matches = list(dict.fromkeys(matches))
        if len(matches) > 1:
            raise DataValidationError(
                f"Multiple columns could be {canonical!r}: "
                + ", ".join(repr(m) for m in matches)
                + ". Pass an explicit columns mapping to disambiguate."
            )
        if matches:
            mapping[canonical] = matches[0]

    for canonical, header in (overrides or {}).items():
        mapping[canonical] = header
    return mapping


@dataclass(frozen=True)
class Position:
    """One security position in a holdings snapshot.

    Attributes
    ----------
    account
        Account label exactly as the exporting system reports it.
    ticker
        Instrument identifier exactly as the exporting system reports it.
    quantity
        Units held. Must be strictly positive.
    price
        Price per unit in *currency* at the snapshot date.
    currency
        ISO 4217 code of *price*. Defaults to ``"CAD"``; case-insensitive.
    cost_basis_cad
        CAD cost basis when the export provides one, otherwise ``None``.
        ``None`` means *unknown* and is never treated as zero.

    Raises
    ------
    DataValidationError
        If *quantity* or *price* is not strictly positive, or *ticker* or
        *account* is blank.
    """

    account: str
    ticker: str
    quantity: float
    price: float
    currency: str = _CAD
    cost_basis_cad: float | None = None

    def __post_init__(self) -> None:
        if not self.account.strip():
            raise DataValidationError("Position account must be non-empty.")
        if not self.ticker.strip():
            raise DataValidationError(
                f"Position ticker must be non-empty (account {self.account!r})."
            )
        if self.quantity <= 0:
            raise DataValidationError(
                f"Position quantity must be strictly positive; got {self.quantity} "
                f"for {self.ticker} in {self.account}."
            )
        if self.price <= 0:
            raise DataValidationError(
                f"Position price must be strictly positive; got {self.price} "
                f"for {self.ticker} in {self.account}."
            )
        code = self.currency.strip().upper()
        if len(code) != 3 or not code.isalpha():
            raise DataValidationError(
                f"Currency must be an ISO 4217 code; got {self.currency!r} for "
                f"{self.ticker} in {self.account}."
            )
        object.__setattr__(self, "currency", code)


@dataclass(frozen=True)
class HoldingsSnapshot:
    """A point-in-time view of accounts, positions, and cash.

    Attributes
    ----------
    as_of
        Date the snapshot describes. Caller-supplied; never inferred.
    positions
        Security positions, at most one per ``(account, ticker)``.
    cash
        ``{account: {currency: amount}}`` settled cash. Amounts are not
        converted until valuation.
    fx_rates
        ``{currency: CAD per unit}``. ``"CAD"`` is always 1.0 and does not
        need an entry.

    Raises
    ------
    DataValidationError
        If two positions share the same ``(account, ticker)``, or a cash
        amount, FX rate, or account label is invalid.
    """

    as_of: date
    positions: tuple[Position, ...]
    cash: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    fx_rates: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.as_of, date) or isinstance(self.as_of, bool):
            raise DataValidationError(
                f"as_of must be a datetime.date; got {type(self.as_of).__name__}."
            )

        seen: set[tuple[str, str]] = set()
        duplicates: list[str] = []
        for position in self.positions:
            key = (position.account, position.ticker)
            if key in seen:
                duplicates.append(f"{position.ticker} in {position.account}")
            seen.add(key)
        if duplicates:
            raise DataValidationError(
                "Duplicate (account, ticker) positions: "
                + ", ".join(sorted(duplicates))
                + ". Aggregate them before importing; PySharpe never merges "
                "unknown-identity rows."
            )

        for currency, rate in self.fx_rates.items():
            if rate <= 0:
                raise DataValidationError(
                    f"FX rate for {currency!r} must be strictly positive; got {rate}."
                )

        for account in self.cash:
            if not account.strip():
                raise DataValidationError("Cash account label must be non-empty.")
            for currency, amount in self.cash[account].items():
                if amount < 0:
                    raise DataValidationError(
                        f"Cash balance for {currency!r} in {account!r} must be "
                        f"non-negative; got {amount}."
                    )

        # A snapshot that cannot be valued in CAD is not a usable snapshot, so
        # FX coverage is enforced here rather than deferred to first valuation.
        for position in self.positions:
            self.rate(position.currency)
        for balances in self.cash.values():
            for currency in balances:
                self.rate(currency)

    # -- Valuation -----------------------------------------------------------

    def rate(self, currency: str) -> float:
        """Return the CAD-per-unit rate for *currency*.

        Raises
        ------
        DataValidationError
            If a non-CAD currency has no supplied rate. Rates are never
            inferred or backfilled.
        """
        code = currency.strip().upper()
        if code == _CAD:
            return 1.0
        try:
            return float(self.fx_rates[code])
        except KeyError:
            affected = sorted(
                f"{p.ticker} in {p.account}"
                for p in self.positions
                if p.currency == code
            )
            detail = f" Affected positions: {', '.join(affected)}." if affected else ""
            raise DataValidationError(
                f"No CAD rate supplied for currency {code!r}; add it to "
                f"fx_rates.{detail} Coverage is explicit: PySharpe does not "
                f"backfill FX rates."
            ) from None

    def position_value_cad(self, position: Position) -> float:
        """CAD market value of *position* at the snapshot's own price and FX."""
        return position.quantity * position.price * self.rate(position.currency)

    def cash_value_cad(self, account: str | None = None) -> float:
        """CAD value of settled cash, for one account or across the snapshot."""
        accounts = [account] if account is not None else list(self.cash)
        return sum(
            amount * self.rate(currency)
            for name in accounts
            for currency, amount in self.cash.get(name, {}).items()
        )

    def securities_value_cad(self) -> float:
        """CAD value of all security positions."""
        return sum(self.position_value_cad(position) for position in self.positions)

    def total_value_cad(self) -> float:
        """CAD value of securities plus cash."""
        return self.securities_value_cad() + self.cash_value_cad()

    # -- Known/unknown cost basis -------------------------------------------

    def missing_cost_basis(self) -> tuple[Position, ...]:
        """Return the positions whose CAD cost basis the export did not supply."""
        return tuple(p for p in self.positions if p.cost_basis_cad is None)

    # -- Downstream adapters -------------------------------------------------

    def current_balances(self) -> pd.DataFrame:
        """Holdings table in the multi-account allocator's schema.

        Columns are ``ticker``, ``account``, ``current_value`` in CAD, matching
        :func:`pysharpe.execution.cash_flow_rebalance.allocate_contribution_cash_flow`.
        """
        return pd.DataFrame(
            [
                {
                    "ticker": position.ticker,
                    "account": position.account,
                    "current_value": self.position_value_cad(position),
                }
                for position in self.positions
            ],
            columns=["ticker", "account", "current_value"],
        )

    def holdings_frame(self) -> pd.DataFrame:
        """Holdings table in the ``--holdings-csv`` schema.

        Columns are ``ticker``, ``account``, ``shares``, ``current_value``
        (CAD) and ``currency``. ``current_value`` plus ``shares`` is the pair
        :func:`pysharpe.execution.rebalance.build_rebalance_plan` accepts.
        """
        return pd.DataFrame(
            [
                {
                    "ticker": position.ticker,
                    "account": position.account,
                    "shares": position.quantity,
                    "current_value": self.position_value_cad(position),
                    "currency": position.currency,
                }
                for position in self.positions
            ],
            columns=["ticker", "account", "shares", "current_value", "currency"],
        )


# ---------------------------------------------------------------------------
# CSV import
# ---------------------------------------------------------------------------


def _is_blank(raw: object) -> bool:
    """Return ``True`` when a CSV cell carries no text."""
    return ("" if raw is None else str(raw).strip()) == ""


def _parse_number(raw: object, *, field_name: str, row: int) -> float:
    """Convert *raw* to a float or raise a row-specific validation error."""
    text = "" if raw is None else str(raw).strip().replace(",", "")
    if text == "":
        raise DataValidationError(
            f"Row {row}: {field_name} is empty; every imported position needs a value."
        )
    try:
        return float(text)
    except ValueError:
        raise DataValidationError(
            f"Row {row}: {field_name} is not numeric (got {raw!r})."
        ) from None


def read_holdings_snapshot(
    path: Path | str,
    *,
    columns: Mapping[str, str] | None = None,
    ticker_transform: Callable[[str], str] | None = None,
    as_of: date,
    fx_rates: Mapping[str, float],
    cash: Mapping[str, Mapping[str, float]] | None = None,
) -> HoldingsSnapshot:
    """Read a holdings export into a :class:`HoldingsSnapshot`.

    Any export works: pass a ``columns`` mapping from the canonical names to
    the file's own headers, for example::

        columns={
            "ticker": "Instrument",
            "account": "Holding Account",
            "quantity": "Units Held",
            "price": "Current Price",
            "currency": "Currency",
            "cost_basis": "Cost Basis",
        }

    Omit ``columns`` to let :func:`infer_column_mapping` match the header row by
    synonym. Pass ``ticker_transform=normalise_ticker`` to convert export
    market codes (``VFV.TSE``, ``VOO.NYSE``) into the tickers PySharpe prices
    with; without it, identifiers are imported exactly as exported.

    Parameters
    ----------
    path
        CSV file to read. UTF-8, comma separated, one row per position.
    columns
        Canonical name -> source header. ``REQUIRED_COLUMNS`` must all be
        mapped; ``cost_basis`` is optional. ``None`` infers the mapping.
    ticker_transform
        Optional callable applied to every ticker before the position is
        created. It must return a non-empty identifier; the result is what the
        snapshot reports and what duplicate detection uses.
    as_of
        Date the export describes.
    fx_rates
        ``{currency: CAD per unit}`` for every non-CAD currency held.
    cash
        Optional ``{account: {currency: amount}}`` settled cash, held
        separately from the file because exports rarely include it in the
        same rows.

    Returns
    -------
    HoldingsSnapshot
        Parsed and validated snapshot.

    Raises
    ------
    DataValidationError
        If a required canonical column cannot be mapped, a row is malformed, or
        the resulting snapshot is inconsistent (duplicate positions,
        non-positive quantities, uncovered currency).
    DataIngestionError
        If the file is missing, empty, or not parseable as CSV.
    """
    source = Path(path)
    csv_path = source

    try:
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        raise DataIngestionError(f"Holdings file not found: {csv_path}") from None
    except pd.errors.EmptyDataError:
        raise DataIngestionError(f"Holdings file is empty: {csv_path}") from None
    except (UnicodeDecodeError, OSError) as exc:
        raise DataIngestionError(
            f"Cannot read holdings file: {csv_path}\nDetails: {exc}"
        ) from exc

    if columns is None:
        columns = infer_column_mapping(list(frame.columns))
        logger.info(
            "Inferred holdings columns from %s: %s", csv_path.name, dict(columns)
        )
    else:
        unknown = set(columns) - set(REQUIRED_COLUMNS) - set(OPTIONAL_COLUMNS)
        if unknown:
            raise DataValidationError(
                "Unknown canonical column names in mapping: "
                + ", ".join(sorted(unknown))
                + f". Allowed: {', '.join(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)}."
            )

    missing_canonical = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing_canonical:
        raise DataValidationError(
            "Cannot map required canonical columns: "
            + ", ".join(missing_canonical)
            + f". File headers: {', '.join(map(str, frame.columns))}. "
            "Pass an explicit columns mapping (canonical name -> header)."
        )

    missing_headers = [
        f"{name!r} <- {header!r}"
        for name, header in columns.items()
        if header not in frame.columns
    ]
    if missing_headers:
        raise DataValidationError(
            f"Mapped columns not found in {csv_path.name}: "
            + ", ".join(missing_headers)
            + f". File headers: {', '.join(map(str, frame.columns))}."
        )

    positions: list[Position] = []
    renames: dict[str, str] = {}
    for offset, row in enumerate(frame.to_dict(orient="records"), start=1):
        record: dict[str, object] = dict(row)  # type: ignore[arg-type]
        quantity_raw = record[columns["quantity"]]
        price_raw = record[columns["price"]]
        if _is_blank(quantity_raw) and _is_blank(price_raw):
            # Group headers, subtotal rows, and footers carry value/gain
            # columns but never both units and a price.
            continue

        ticker = str(record[columns["ticker"]]).strip()
        account = str(record[columns["account"]]).strip()
        if ticker_transform is not None and ticker:
            transformed = str(ticker_transform(ticker)).strip()
            if not transformed:
                raise DataValidationError(
                    f"Row {offset}: ticker_transform returned an empty ticker "
                    f"for {ticker!r}."
                )
            if transformed != ticker:
                renames[ticker] = transformed
            ticker = transformed

        quantity = _parse_number(quantity_raw, field_name="quantity", row=offset)
        price = _parse_number(price_raw, field_name="price", row=offset)
        currency = str(record[columns["currency"]]).strip() or _CAD

        cost_basis_cad: float | None = None
        if "cost_basis" in columns:
            raw_basis = str(record[columns["cost_basis"]]).strip().replace(",", "")
            if raw_basis != "":
                cost_basis_cad = _parse_number(
                    raw_basis, field_name="cost_basis", row=offset
                )

        try:
            positions.append(
                Position(
                    account=account,
                    ticker=ticker,
                    quantity=quantity,
                    price=price,
                    currency=currency,
                    cost_basis_cad=cost_basis_cad,
                )
            )
        except DataValidationError as exc:
            raise DataValidationError(f"Row {offset}: {exc}") from None

    if not positions:
        raise DataValidationError(
            f"Holdings file {csv_path.name} contains no positions. Export at "
            f"least one held security before importing."
        )

    logger.info(
        "Imported %d positions from %s as of %s",
        len(positions),
        csv_path.name,
        as_of.isoformat(),
    )
    if renames:
        logger.info(
            "Normalised %d ticker(s): %s",
            len(renames),
            ", ".join(f"{raw} -> {new}" for raw, new in sorted(renames.items())),
        )
    return HoldingsSnapshot(
        as_of=as_of,
        positions=tuple(positions),
        cash=cash or {},
        fx_rates=fx_rates,
    )


__all__ = [
    "MARKET_SUFFIXES",
    "HoldingsSnapshot",
    "OPTIONAL_COLUMNS",
    "Position",
    "REQUIRED_COLUMNS",
    "infer_column_mapping",
    "normalise_ticker",
    "read_holdings_snapshot",
]
