"""Build contribution plans from saved optimisation artefacts."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from pysharpe.config import (
    ExecutionConfig,
    get_settings,
    get_ticker_metadata,
)
from pysharpe.exceptions import (
    DataIngestionError,
    DataValidationError,
    PySharpeError,
)
from pysharpe.optimization.tax_location import (
    AssetLocationEngine,
    AssetTaxCharacteristics,
    TaxProfile,
)

from .allocator import (
    AllocationConfig,
    allocate_contribution,
    score_opportunities,
)

if TYPE_CHECKING:
    from pysharpe.data.interop import HoldingsSnapshot
    from pysharpe.execution.ledger import ReconciliationReport

logger = logging.getLogger(__name__)

_TICKER_ALIASES = {"ticker", "symbol"}
_VALUE_ALIASES = {
    "amount",
    "current_value",
    "dollar_value",
    "market_value",
    "total_value",
    "value",
}
_SHARE_ALIASES = {"quantity", "share_count", "shares", "units"}
_VALUATION_COLUMNS = ("pe_ratio", "pb_ratio", "div_yield", "momentum_6m")
_ACCOUNT_ALIASES = {"account", "account_type", "acct_type", "registration"}


@dataclass(frozen=True)
class RebalancePlan:
    """Represent a fully prepared contribution plan.

    Attributes
    ----------
    portfolio_name : str
        Portfolio identifier used to locate optimisation artefacts.
    new_cash : float
        Dollar amount of fresh capital to deploy.
    weights_path : Path or None
        Path to the saved optimiser output or weights CSV, when file-backed.
    prices_path : Path or None
        Path to the collated/price CSV used for latest prices, when file-backed.
    scored_state : pandas.DataFrame
        Merged portfolio state after :func:`score_opportunities` has appended
        weight drift, valuation, and opportunity columns.
    allocations : pandas.DataFrame
        Final allocation table returned by :func:`allocate_contribution`,
        enriched with latest price, share estimates, and post-buy weights.
    account_allocations : dict[str, pandas.DataFrame] or None
        When the holdings CSV includes an account column, this dictionary maps
        each account label (e.g. ``"TFSA"``, ``"RRSP"``) to its per-account
        allocation DataFrame. ``None`` for single-account plans.
    account_cash : dict[str, float] or None
        Per-account cash deployment amounts. ``None`` for single-account plans.
    """

    portfolio_name: str
    new_cash: float
    weights_path: Path | None
    prices_path: Path | None
    scored_state: pd.DataFrame
    allocations: pd.DataFrame
    account_allocations: dict[str, pd.DataFrame] | None = None
    account_cash: dict[str, float] | None = None

    @property
    def buy_orders(self) -> pd.DataFrame:
        """Return rows with a positive recommended buy.

        Returns
        -------
        pandas.DataFrame
            Subset of ``allocations`` whose ``recommended_allocation`` is
            strictly positive.
        """

        return self.allocations.loc[
            self.allocations["recommended_allocation"] > 0
        ].copy()

    @property
    def is_multi_account(self) -> bool:
        """Return True when the plan includes per-account allocations."""
        return self.account_allocations is not None

    @property
    def accounts(self) -> list[str]:
        """Return the list of account labels, or empty for single-account plans."""
        if self.account_allocations is None:
            return []
        return list(self.account_allocations.keys())


def _normalise_label(label: object) -> str:
    """Convert a user-supplied column label into a canonical lookup key.

    Parameters
    ----------
    label : object
        Raw column label from a CSV or DataFrame.

    Returns
    -------
    str
        Lower-cased label with spaces and hyphens normalized to underscores.
    """

    return str(label).strip().lower().replace("-", "_").replace(" ", "_")


def _rename_known_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common holdings column aliases onto PySharpe's canonical names.

    Parameters
    ----------
    frame : pandas.DataFrame
        Input holdings or weights table.

    Returns
    -------
    pandas.DataFrame
        Copy of ``frame`` with recognized ticker, value, share, and valuation
        columns renamed to the names expected by the rebalance workflow.
    """

    renamed: dict[object, str] = {}
    for column in frame.columns:
        label = _normalise_label(column)
        if label in _TICKER_ALIASES:
            renamed[column] = "ticker"
        elif label in _VALUE_ALIASES:
            renamed[column] = "current_value"
        elif label in _SHARE_ALIASES:
            renamed[column] = "shares"
        elif label in _VALUATION_COLUMNS:
            renamed[column] = label
        elif label in _ACCOUNT_ALIASES:
            renamed[column] = "account"
    return frame.rename(columns=renamed)


def _coerce_non_negative(
    frame: pd.DataFrame,
    column: str,
    *,
    label: str,
) -> pd.Series:
    """Convert a column to numeric values and reject invalid or negative rows.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table containing the column to validate.
    column : str
        Column name to coerce.
    label : str
        Human-readable label used in validation errors.

    Returns
    -------
    pandas.Series
        Numeric series aligned to ``frame.index``.

    Raises
    ------
    ValueError
        If any value is missing after coercion or is negative.
    """

    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = frame.loc[values.isna(), "ticker"].tolist()
    if invalid:
        raise DataValidationError(
            f"Non-numeric {label} values for: {', '.join(invalid)}. "
            f"Check that the '{column}' column contains only numbers."
        )

    negative = frame.loc[values < 0, "ticker"].tolist()
    if negative:
        raise DataValidationError(
            f"{label.capitalize()} values cannot be negative: {', '.join(negative)}. "
            f"Verify the '{column}' column in your input data."
        )

    return values


def _prepare_ticker_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the ticker column for downstream merges.

    Parameters
    ----------
    frame : pandas.DataFrame
        Input table expected to contain a ticker column.

    Returns
    -------
    pandas.DataFrame
        Copy of ``frame`` with stripped non-empty ticker symbols.

    Raises
    ------
    ValueError
        If the ticker column is missing or the cleaned frame becomes empty.
    """

    if "ticker" not in frame.columns:
        raise DataValidationError(
            "Input data is missing a ticker column. "
            f"Found columns: {', '.join(str(c) for c in frame.columns)}."
        )

    prepared = frame.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.strip()
    prepared = prepared.loc[prepared["ticker"] != ""].copy()
    if prepared.empty:
        raise DataValidationError(
            "Holdings data contains no valid ticker entries. "
            "Ensure at least one row has a non-blank ticker symbol."
        )
    return prepared


def _parse_target_weights_csv(
    path: Path,
    *,
    portfolio_name: str,
) -> tuple[pd.DataFrame, Path]:
    """Load target weights from a CSV/TXT file."""
    try:
        frame = _rename_known_columns(pd.read_csv(path))
    except pd.errors.EmptyDataError:
        raise DataIngestionError(
            f"Weights file for '{portfolio_name}' is empty: {path}\n"
            f"Re-run 'pysharpe optimise --portfolio {portfolio_name}' to "
            f"regenerate it."
        ) from None
    except pd.errors.ParserError as exc:
        raise DataIngestionError(
            f"Unable to parse weights file for '{portfolio_name}': {path}\n"
            f"The file may be corrupted or use an unexpected format. "
            f"Details: {exc}"
        ) from exc
    except (UnicodeDecodeError, OSError) as exc:
        raise DataIngestionError(
            f"Cannot read weights file for '{portfolio_name}': {path}\nDetails: {exc}"
        ) from exc

    if "ticker" not in frame.columns:
        raise DataValidationError(
            f"Weights file for '{portfolio_name}' is missing a ticker column. "
            f"Found columns: {', '.join(str(c) for c in frame.columns)}. "
            f"Path: {path}"
        )
    if "weight" not in frame.columns and "target_weight" not in frame.columns:
        raise DataValidationError(
            f"Weights file for '{portfolio_name}' is missing a weight column. "
            f"Found columns: {', '.join(str(c) for c in frame.columns)}. "
            f"Path: {path}"
        )

    if "target_weight" not in frame.columns:
        frame = frame.rename(columns={"weight": "target_weight"})

    frame = _prepare_ticker_frame(frame)
    frame["target_weight"] = pd.to_numeric(
        frame["target_weight"], errors="coerce"
    ).fillna(0.0)
    return frame[["ticker", "target_weight"]], path


def _load_target_weights(
    portfolio_name: str,
    export_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    """Load target weights emitted by the optimisation workflow."""
    path = export_dir / f"{portfolio_name}_weights.txt"
    if not path.exists():
        raise DataIngestionError(
            f"Optimisation weights file not found for portfolio "
            f"'{portfolio_name}' at: {path}\n"
            f"Run 'pysharpe optimise --portfolio {portfolio_name}' first to "
            f"generate the weights."
        )
    return _parse_target_weights_csv(path, portfolio_name=portfolio_name)


def _resolve_target_weights(
    portfolio_name: str,
    export_dir: Path,
    target_weights: Mapping[str, float] | Path | None,
) -> tuple[pd.DataFrame, Path | None]:
    """Resolve target weights from an explicit mapping, path, or export directory."""
    if target_weights is None:
        return _load_target_weights(portfolio_name, export_dir)
    if isinstance(target_weights, Path):
        resolved = target_weights.expanduser().resolve()
        if not resolved.exists():
            raise DataIngestionError(f"Weights file not found: {resolved}")
        return _parse_target_weights_csv(resolved, portfolio_name=portfolio_name)

    rows: list[dict[str, object]] = []
    for raw_ticker, raw_weight in target_weights.items():
        clean = str(raw_ticker).strip()
        if not clean:
            raise DataValidationError("Target weights contain an empty ticker symbol.")
        val = float(raw_weight)
        if not np.isfinite(val) or val < 0.0:
            raise DataValidationError(
                f"Target weight for '{clean}' must be finite and non-negative; got {raw_weight!r}."
            )
        rows.append({"ticker": clean, "target_weight": val})

    if not rows:
        raise DataValidationError("Target weights mapping cannot be empty.")

    frame = (
        pd.DataFrame(rows)
        .groupby("ticker", as_index=False)
        .agg({"target_weight": "sum"})
    )
    if float(frame["target_weight"].sum()) <= 0.0:
        raise DataValidationError("Total target weight must be strictly positive.")
    return pd.DataFrame(frame.loc[:, ["ticker", "target_weight"]]), None


def _parse_latest_prices_csv(
    path: Path,
    *,
    portfolio_name: str,
) -> tuple[pd.DataFrame, Path]:
    """Extract latest non-null prices from a collated history or ticker/price CSV."""
    try:
        header_probe = _rename_known_columns(pd.read_csv(path, nrows=0))
        if "ticker" in header_probe.columns and (
            "price" in header_probe.columns or "latest_price" in header_probe.columns
        ):
            table = _rename_known_columns(pd.read_csv(path))
            price_col = "latest_price" if "latest_price" in table.columns else "price"
            table = _prepare_ticker_frame(table)
            table["latest_price"] = pd.to_numeric(table[price_col], errors="coerce")
            table = table.dropna(subset=["latest_price"])
            invalid = table[table["latest_price"] <= 0]
            if not invalid.empty:
                bad = ", ".join(invalid["ticker"].astype(str).tolist())
                raise DataValidationError(
                    f"Latest prices must be strictly positive for: {bad}"
                )
            if table.empty:
                raise DataIngestionError(
                    f"No valid positive prices found in price file: {path}"
                )
            return table[["ticker", "latest_price"]].drop_duplicates(
                subset=["ticker"], keep="last"
            ), path
        frame = pd.read_csv(path, index_col=0)
    except DataIngestionError:
        raise
    except DataValidationError:
        raise
    except pd.errors.EmptyDataError:
        raise DataIngestionError(
            f"Collated price history for '{portfolio_name}' is empty: {path}"
        ) from None
    except pd.errors.ParserError as exc:
        raise DataIngestionError(
            f"Unable to parse collated price history for "
            f"'{portfolio_name}': {path}\n"
            f"The file may be corrupted or use an unexpected format. "
            f"Details: {exc}"
        ) from exc
    except (UnicodeDecodeError, OSError) as exc:
        raise DataIngestionError(
            f"Cannot read collated price history for "
            f"'{portfolio_name}': {path}\n"
            f"Details: {exc}"
        ) from exc

    if frame.empty:
        raise DataIngestionError(
            f"Collated price history for '{portfolio_name}' is empty: {path}"
        )

    prices = frame.apply(pd.to_numeric, errors="coerce").ffill()
    if prices.empty:
        raise DataIngestionError(
            f"Collated price history for '{portfolio_name}' has no numeric "
            f"price columns: {path}"
        )

    latest = prices.iloc[-1].dropna()
    if latest.empty:
        raise DataIngestionError(
            f"Unable to derive latest prices for '{portfolio_name}' from "
            f"{path}. All rows may contain NaN values."
        )

    latest_prices = latest.rename_axis("ticker").reset_index(name="latest_price")
    latest_prices["ticker"] = latest_prices["ticker"].astype(str).str.strip()
    return latest_prices, path


def _load_latest_prices(
    portfolio_name: str,
    export_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    """Extract the most recent non-null prices from a collated history file."""
    path = export_dir / f"{portfolio_name}_collated.csv"
    if not path.exists():
        raise DataIngestionError(
            f"Collated price history not found for portfolio "
            f"'{portfolio_name}' at: {path}\n"
            f"Run 'pysharpe optimise --portfolio {portfolio_name}' first to "
            f"generate it."
        )
    return _parse_latest_prices_csv(path, portfolio_name=portfolio_name)


def _record_consistent_cad_price(
    prices_by_ticker: dict[str, float],
    ticker: str,
    price_cad: float,
) -> None:
    """Record a ticker's CAD price, rejecting contradictory quotes across accounts."""
    if not np.isfinite(price_cad) or price_cad <= 0.0:
        raise DataValidationError(
            f"Derived CAD price for '{ticker}' must be finite and positive; got {price_cad!r}."
        )
    existing = prices_by_ticker.get(ticker)
    if existing is not None and not np.isclose(
        existing, price_cad, rtol=1e-6, atol=1e-6
    ):
        raise DataValidationError(
            f"Conflicting CAD unit prices for '{ticker}' across accounts: "
            f"{existing:.6f} vs {price_cad:.6f}. Supply explicit latest_prices to resolve."
        )
    prices_by_ticker[ticker] = float(price_cad)


def _extract_prices_from_source(
    *,
    snapshot: HoldingsSnapshot | None,
    reconciliation: ReconciliationReport | None,
) -> pd.DataFrame:
    """Derive per-ticker CAD latest prices from a snapshot or reconciliation report."""
    prices_by_ticker: dict[str, float] = {}
    if snapshot is not None:
        for position in snapshot.positions:
            cad_price = float(position.price) * snapshot.rate(position.currency)
            _record_consistent_cad_price(prices_by_ticker, position.ticker, cad_price)
    elif reconciliation is not None:
        for summary in reconciliation.accounts.values():
            for holding in summary.holdings:
                if holding.shares > 0:
                    cad_price = float(holding.market_value_cad / holding.shares)
                    _record_consistent_cad_price(
                        prices_by_ticker, holding.ticker, cad_price
                    )

    if not prices_by_ticker:
        return pd.DataFrame(columns=["ticker", "latest_price"])
    return pd.DataFrame(
        [
            {"ticker": ticker, "latest_price": price}
            for ticker, price in sorted(prices_by_ticker.items())
        ],
        columns=["ticker", "latest_price"],
    )


def _resolve_latest_prices(
    portfolio_name: str,
    export_dir: Path,
    latest_prices: Mapping[str, float] | Path | None,
    *,
    export_dir_explicit: bool,
    snapshot: HoldingsSnapshot | None,
    reconciliation: ReconciliationReport | None,
    target_tickers: set[str],
) -> tuple[pd.DataFrame, Path | None]:
    """Resolve CAD latest prices from explicit input, snapshot/ledger, or exports."""
    if isinstance(latest_prices, Path):
        resolved = latest_prices.expanduser().resolve()
        if not resolved.exists():
            raise DataIngestionError(f"Latest prices file not found: {resolved}")
        return _parse_latest_prices_csv(resolved, portfolio_name=portfolio_name)

    if latest_prices is not None:
        rows: list[dict[str, object]] = []
        for raw_ticker, raw_price in latest_prices.items():
            clean = str(raw_ticker).strip()
            if not clean:
                raise DataValidationError(
                    "latest_prices contains an empty ticker symbol."
                )
            val = float(raw_price)
            if not np.isfinite(val) or val <= 0.0:
                raise DataValidationError(
                    f"Latest price for '{clean}' must be finite and strictly positive; got {raw_price!r}."
                )
            rows.append({"ticker": clean, "latest_price": val})
        if not rows:
            raise DataValidationError("latest_prices mapping cannot be empty.")
        frame = pd.DataFrame(rows).drop_duplicates(subset=["ticker"], keep="last")
        return frame[["ticker", "latest_price"]], None

    if snapshot is None and reconciliation is None:
        return _load_latest_prices(portfolio_name, export_dir)

    derived = _extract_prices_from_source(
        snapshot=snapshot, reconciliation=reconciliation
    )
    covered = set(derived["ticker"].tolist())
    missing = sorted(target_tickers - covered)
    prices_path: Path | None = None

    if missing and export_dir_explicit:
        collated_path = export_dir / f"{portfolio_name}_collated.csv"
        if collated_path.exists():
            file_prices, prices_path = _parse_latest_prices_csv(
                collated_path, portfolio_name=portfolio_name
            )
            extra = file_prices[~file_prices["ticker"].isin(covered)]
            derived = pd.concat([derived, extra], ignore_index=True)
            covered = set(derived["ticker"].tolist())
            missing = sorted(target_tickers - covered)

    if missing:
        raise DataValidationError(
            "Missing CAD latest price for target ticker(s): "
            + ", ".join(missing)
            + ". Provide latest_prices or a collated CSV covering them."
        )
    return derived, prices_path


def _parse_json_payload(
    raw: str,
    *,
    label: str,
    read_error_prefix: str | None = None,
    json_error_detail: str | None = None,
) -> object:
    """Read an inline JSON string or JSON file path and decode it."""
    candidate = Path(raw).expanduser()
    try:
        is_file = candidate.exists()
    except OSError:
        is_file = False

    if is_file:
        try:
            payload = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            prefix = read_error_prefix or f"Cannot read {label.lower()} file"
            raise DataIngestionError(f"{prefix}: {candidate}\nDetails: {exc}") from exc
    else:
        payload = raw

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        if json_error_detail is not None:
            raise DataValidationError(json_error_detail.format(exc=exc)) from exc
        raise DataValidationError(
            f"{label} JSON is not valid.\nDetails: {exc}"
        ) from exc


def _parse_numeric_json_mapping(
    raw: str,
    *,
    label: str,
    allow_zero: bool = True,
    read_error_prefix: str | None = None,
    json_error_detail: str | None = None,
    non_dict_message: str | None = None,
    invalid_value_template: str | None = None,
    negative_value_template: str | None = None,
    empty_message: str | None = None,
) -> dict[str, float]:
    """Parse inline JSON or a JSON file path into a string-to-float mapping."""
    data = _parse_json_payload(
        raw,
        label=label,
        read_error_prefix=read_error_prefix,
        json_error_detail=json_error_detail,
    )
    if not isinstance(data, dict):
        raise DataValidationError(
            non_dict_message
            or f"{label} JSON must be an object mapping keys to numbers."
        )

    parsed: dict[str, float] = {}
    for key, value in data.items():
        clean_key = str(key).strip()
        if not clean_key:
            continue

        if isinstance(value, bool):
            numeric = float("nan")
        else:
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

        if pd.isna(numeric) or not np.isfinite(float(numeric)):
            template = (
                invalid_value_template
                or f"Invalid {label.lower()} value for '{{ticker}}'. Value must be a number, got: {{value!r}}"
            )
            raise DataValidationError(template.format(ticker=clean_key, value=value))

        float_val = float(numeric)
        if float_val < 0 or (not allow_zero and float_val == 0):
            if float_val < 0 and negative_value_template:
                raise DataValidationError(
                    negative_value_template.format(ticker=clean_key, numeric=float_val)
                )
            bound = "non-negative" if allow_zero else "strictly positive"
            raise DataValidationError(
                f"{label} value for '{clean_key}' must be {bound}; got: {float_val}"
            )
        parsed[clean_key] = float_val

    if not parsed:
        raise DataValidationError(
            empty_message or f"{label} JSON did not contain any valid entries."
        )
    return parsed


def _parse_cash_json(raw: str) -> dict[str, dict[str, float]]:
    """Parse inline JSON or file path mapping account -> amount or currency map."""
    data = _parse_json_payload(raw, label="Cash")
    if not isinstance(data, dict):
        raise DataValidationError(
            "Cash JSON must map each account to a CAD number or {currency: amount}."
        )

    parsed: dict[str, dict[str, float]] = {}
    for account, entry in data.items():
        clean_account = str(account).strip()
        if not clean_account:
            raise DataValidationError("Cash JSON contains an empty account label.")
        if isinstance(entry, dict):
            ccy_map: dict[str, float] = {}
            for raw_ccy, raw_amt in entry.items():
                ccy = str(raw_ccy).strip().upper()
                if not ccy:
                    raise DataValidationError(
                        f"Cash entry for account '{clean_account}' contains an empty currency code."
                    )
                if isinstance(raw_amt, bool):
                    raise DataValidationError(
                        f"Cash amount for '{clean_account}' ({ccy}) must be numeric; got {raw_amt!r}."
                    )
                try:
                    amt = float(raw_amt)
                except (TypeError, ValueError) as exc:
                    raise DataValidationError(
                        f"Cash amount for '{clean_account}' ({ccy}) must be numeric; got {raw_amt!r}."
                    ) from exc
                if not np.isfinite(amt) or amt < 0:
                    raise DataValidationError(
                        f"Cash amount for '{clean_account}' ({ccy}) must be finite and non-negative; got {raw_amt!r}."
                    )
                ccy_map[ccy] = amt
            parsed[clean_account] = ccy_map
        elif isinstance(entry, (int, float)) and not isinstance(entry, bool):
            amt = float(entry)
            if not np.isfinite(amt) or amt < 0:
                raise DataValidationError(
                    f"Cash amount for '{clean_account}' must be finite and non-negative; got {entry!r}."
                )
            parsed[clean_account] = {"CAD": amt}
        else:
            raise DataValidationError(
                f"Cash entry for '{clean_account}' must be a number or {{currency: amount}} object; got {type(entry).__name__}."
            )
    return parsed


def _parse_account_types_json(raw: str) -> dict[str, str]:
    """Parse inline JSON or file path mapping account labels to tax account types."""
    data = _parse_json_payload(raw, label="Account types")
    if not isinstance(data, dict) or not data:
        raise DataValidationError(
            "Account types JSON must be a non-empty object mapping account label to account type."
        )
    parsed: dict[str, str] = {}
    for acct, acct_type in data.items():
        clean_acct = str(acct).strip()
        clean_type = str(acct_type).strip()
        if not clean_acct or not clean_type:
            raise DataValidationError(
                "Account types JSON entries must have non-empty account labels and types."
            )
        parsed[clean_acct] = clean_type
    return parsed


def _load_holdings_mapping(raw: str) -> dict[str, float]:
    """Parse inline JSON or a JSON file path into a holdings mapping."""
    return _parse_numeric_json_mapping(
        raw,
        label="Holdings",
        allow_zero=True,
        read_error_prefix="Cannot read holdings file",
        json_error_detail=(
            "Holdings JSON is not valid. Check for missing commas, "
            "trailing commas, or unquoted keys.\n"
            "Details: {exc}"
        ),
        non_dict_message="Holdings JSON must be an object mapping ticker to value.",
        invalid_value_template=(
            "Invalid holdings value for ticker '{ticker}'. "
            "Value must be a number, got: {value!r}"
        ),
        negative_value_template=(
            "Holdings value cannot be negative for ticker '{ticker}'. Got: {numeric}"
        ),
        empty_message=(
            "Holdings JSON did not contain any valid ticker entries. "
            "Provide at least one ticker with a numeric value."
        ),
    )


def _holdings_from_snapshot(
    snapshot: HoldingsSnapshot,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Convert a :class:`HoldingsSnapshot` into CAD holdings and cash maps."""
    raw_frame = snapshot.holdings_frame()
    if raw_frame.empty:
        frame = pd.DataFrame(
            columns=["ticker", "account", "current_value", "current_shares"]
        )
    else:
        frame = raw_frame.rename(columns={"shares": "current_shares"})[
            ["ticker", "account", "current_value", "current_shares"]
        ].copy()
        frame["ticker"] = frame["ticker"].astype(str).str.strip()
        frame["account"] = frame["account"].astype(str).str.strip()
        frame = (
            frame.groupby(["ticker", "account"], as_index=False)[
                ["current_value", "current_shares"]
            ]
            .sum()
            .sort_values(["account", "ticker"])
            .reset_index(drop=True)
        )

    cash_by_account: dict[str, float] = {}
    for acct, balances in snapshot.cash.items():
        clean_acct = str(acct).strip()
        if not clean_acct:
            raise DataValidationError(
                "HoldingsSnapshot cash contains an empty account label."
            )
        total_cad = 0.0
        for ccy, amt in balances.items():
            total_cad += float(amt) * snapshot.rate(ccy)
        cash_by_account[clean_acct] = round(
            cash_by_account.get(clean_acct, 0.0) + total_cad, 2
        )

    return frame, cash_by_account


def _holdings_from_reconciliation(
    reconciliation: ReconciliationReport,
    *,
    account_by: Literal["account_type", "account_id"],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Convert a :class:`ReconciliationReport` into CAD holdings and cash maps."""
    if account_by not in {"account_type", "account_id"}:
        raise DataValidationError(
            f"account_by must be 'account_type' or 'account_id'; got {account_by!r}."
        )
    frame = reconciliation.to_holdings_frame(by=account_by).copy()
    if not frame.empty:
        frame = (
            frame.groupby(["ticker", "account"], as_index=False)[
                ["current_value", "current_shares"]
            ]
            .sum()
            .sort_values(["account", "ticker"])
            .reset_index(drop=True)
        )

    cash_by_account: dict[str, float] = {}
    for summary in reconciliation.accounts.values():
        acct_type = (
            summary.account_type.value
            if hasattr(summary.account_type, "value")
            else str(summary.account_type)
        )
        key = acct_type if account_by == "account_type" else summary.account_id
        cash_by_account[key] = round(
            cash_by_account.get(key, 0.0) + float(summary.cash_total_cad),
            2,
        )

    return frame, cash_by_account


def _load_holdings_frame(
    *,
    holdings_csv: Path | None,
    holdings_mapping: Mapping[str, float] | None,
    holdings_kind: Literal["value", "shares"] | None,
    latest_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize current holdings into the allocator's portfolio-state schema.

    Parameters
    ----------
    holdings_csv : pathlib.Path or None
        CSV containing current holdings. Mutually exclusive with
        ``holdings_mapping``.
    holdings_mapping : Mapping[str, float] or None
        In-memory mapping of ticker to current value or share count.
    holdings_kind : {"value", "shares"} or None
        Explicit interpretation of holdings values. When omitted for CSV input,
        the function auto-detects whether the data is value-based or
        share-based.
    latest_prices : pandas.DataFrame
        Latest price lookup used to translate share counts into dollar value.

    Returns
    -------
    pandas.DataFrame
        Holdings table with canonical ticker/current_value columns, optional
        ``current_shares``, and any supplied valuation columns.

    Raises
    ------
    DataValidationError
        If the input is missing required columns, contains invalid numbers, or
        requires prices that are unavailable.
    DataIngestionError
        If a CSV file cannot be read or parsed.
    """

    try:
        if holdings_csv is not None:
            frame = _rename_known_columns(pd.read_csv(holdings_csv))
        elif holdings_mapping is not None:
            if holdings_kind is None:
                raise DataValidationError(
                    "holdings_kind is required when holdings are provided "
                    "as a mapping. Use --holdings-kind value or --holdings-kind shares."
                )
            value_column = "current_value" if holdings_kind == "value" else "shares"
            frame = pd.DataFrame(
                {
                    "ticker": list(holdings_mapping.keys()),
                    value_column: list(holdings_mapping.values()),
                }
            )
        else:
            raise DataValidationError(
                "No holdings provided. Specify --holdings-csv or --holdings-json."
            )
    except DataIngestionError:
        raise
    except DataValidationError:
        raise
    except pd.errors.EmptyDataError:
        raise DataIngestionError(f"Holdings CSV is empty: {holdings_csv}") from None
    except pd.errors.ParserError as exc:
        raise DataIngestionError(
            f"Unable to parse holdings CSV: {holdings_csv}\nDetails: {exc}"
        ) from exc
    except (UnicodeDecodeError, OSError) as exc:
        raise DataIngestionError(
            f"Cannot read holdings CSV: {holdings_csv}\nDetails: {exc}"
        ) from exc

    frame = _prepare_ticker_frame(frame)

    # Preserve account column if present in the raw input
    has_account = "account" in frame.columns
    if has_account:
        account_labels = frame["account"].astype(str).str.strip()
        missing_account = frame["account"].isna() | (account_labels == "")
        if missing_account.any():
            tickers = frame.loc[missing_account, "ticker"].drop_duplicates().tolist()
            raise DataValidationError(
                "Holdings rows are missing an account label for: "
                + ", ".join(tickers)
                + ". Populate the account column or omit it entirely for "
                "single-account mode."
            )
        frame["account"] = account_labels

    if holdings_kind is None:
        if "current_value" in frame.columns:
            resolved_kind: Literal["value", "shares"] = "value"
        elif "shares" in frame.columns:
            resolved_kind = "shares"
        else:
            raise DataValidationError(
                "Holdings CSV must include either a current_value/total_value "
                "column or a shares column."
            )
    else:
        resolved_kind = holdings_kind
        if resolved_kind == "value" and "current_value" not in frame.columns:
            data_columns = [column for column in frame.columns if column != "ticker"]
            if len(data_columns) == 1:
                frame = frame.rename(columns={data_columns[0]: "current_value"})
            else:
                raise DataValidationError(
                    "Holdings input is missing a current_value column."
                )
        if resolved_kind == "shares" and "shares" not in frame.columns:
            data_columns = [column for column in frame.columns if column != "ticker"]
            if len(data_columns) == 1:
                frame = frame.rename(columns={data_columns[0]: "shares"})
            else:
                raise DataValidationError("Holdings input is missing a shares column.")

    account_cols = ["ticker"]
    if has_account:
        account_cols.append("account")
    base = frame[account_cols].copy()
    if resolved_kind == "value":
        base["current_value"] = _coerce_non_negative(
            frame,
            "current_value",
            label="current value",
        )
        if "shares" in frame.columns:
            base["current_shares"] = _coerce_non_negative(
                frame,
                "shares",
                label="shares",
            )
    else:
        base["current_shares"] = _coerce_non_negative(frame, "shares", label="shares")
        price_lookup = latest_prices.set_index("ticker")["latest_price"]
        base["latest_price"] = base["ticker"].map(price_lookup)
        missing_prices = base.loc[base["latest_price"].isna(), "ticker"].tolist()
        if missing_prices:
            raise DataValidationError(
                "Missing latest prices for share-based holdings: "
                + ", ".join(missing_prices)
            )
        base["current_value"] = base["current_shares"] * base["latest_price"]
        base = base.drop(columns=["latest_price"])

    for column in _VALUATION_COLUMNS:
        if column in frame.columns:
            base[column] = pd.to_numeric(frame[column], errors="coerce")

    aggregation: dict[str, str] = {"current_value": "sum"}
    if "current_shares" in base.columns:
        aggregation["current_shares"] = "sum"
    for column in _VALUATION_COLUMNS:
        if column in base.columns:
            aggregation[column] = "first"

    group_keys = ["ticker"]
    if has_account:
        group_keys.append("account")
    return base.groupby(group_keys, as_index=False).agg(aggregation)


def _load_allocation_inputs(
    config_path: Path | None,
) -> tuple[AllocationConfig, pd.DataFrame | None]:
    """Load allocator weights and optional fundamentals from JSON config.

    Parameters
    ----------
    config_path : pathlib.Path or None
        Optional JSON path containing ``allocation_weights`` and
        ``fundamentals`` sections.

    Returns
    -------
    tuple[AllocationConfig, pandas.DataFrame or None]
        Parsed allocator configuration and an optional fundamentals table keyed
        by ticker.

    Raises
    ------
    DataIngestionError
        If ``config_path`` is provided but does not exist or cannot be parsed.
    """

    if config_path is None:
        return AllocationConfig(weight_underweight=0.0, weight_valuation=0.0), None

    path = Path(config_path).expanduser()
    if not path.exists():
        raise DataIngestionError(
            f"Allocation config file not found: {path}\n"
            f"Provide a valid path to a JSON configuration file."
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            config_data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise DataIngestionError(
            f"Allocation config file is not valid JSON: {path}\nDetails: {exc}"
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise DataIngestionError(
            f"Cannot read allocation config file: {path}\nDetails: {exc}"
        ) from exc

    fields = set(AllocationConfig.__dataclass_fields__)
    weights = config_data.get("allocation_weights", {})
    kwargs: dict = {"weight_underweight": 0.0, "weight_valuation": 0.0}
    kwargs.update({key: value for key, value in weights.items() if key in fields})
    config = AllocationConfig(**kwargs)

    fundamentals = config_data.get("fundamentals")
    if not fundamentals:
        return config, None

    frame = pd.DataFrame.from_dict(fundamentals, orient="index").reset_index()
    frame = frame.rename(columns={"index": "ticker"})
    frame = _rename_known_columns(frame)
    frame = _prepare_ticker_frame(frame)

    keep_columns = ["ticker"]
    for column in _VALUATION_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            keep_columns.append(column)

    return config, frame[keep_columns]


def _merge_fundamentals(
    frame: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
) -> pd.DataFrame:
    """Overlay config-level fundamentals onto a merged holdings table.

    Parameters
    ----------
    frame : pandas.DataFrame
        Base merged portfolio state.
    fundamentals : pandas.DataFrame or None
        Optional fundamentals keyed by ticker.

    Returns
    -------
    pandas.DataFrame
        ``frame`` with missing valuation columns filled from ``fundamentals``.
    """

    if fundamentals is None:
        return frame

    merged = frame.merge(fundamentals, on="ticker", how="left", suffixes=("", "_cfg"))
    for column in _VALUATION_COLUMNS:
        config_column = f"{column}_cfg"
        if config_column not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[column].combine_first(merged[config_column])
        else:
            merged[column] = merged[config_column]
        merged = merged.drop(columns=[config_column])
    return merged


def _determine_account_tax_adjustment(
    account: str,
    ticker: str,
    proxy_map: dict[str, dict[str, object]] | None = None,
) -> float:
    """Return a multiplier (0.0–1.0) applied to opportunity scores for tax drag.

    US-domiciled assets held in TFSA or FHSA accounts lose ~15% of dividend
    yield to withholding tax. This function returns a penalty factor that
    reduces the opportunity score proportionally, pushing those assets toward
    RRSP during allocation.

    Parameters
    ----------
    account : str
        Account type label (TFSA, RRSP, FHSA, Non-Reg, etc.).
    ticker : str
        Ticker symbol to check for US domicile.
    proxy_map : dict or None
        Proxy metadata used to determine US domicile per ticker.

    Returns
    -------
    float
        A multiplier in [0.8, 1.0]. 1.0 means no penalty; lower values
        penalize the opportunity score for tax-disadvantaged placements.
    """
    meta = get_ticker_metadata(ticker, proxy_map=proxy_map)
    if not meta["is_us_domiciled"]:
        return 1.0

    account_upper = account.upper().strip()
    # TFSA and FHSA have no treaty protection; US withholding tax applies
    if account_upper in {"TFSA", "FHSA"}:
        # 15% withholding on ~2% dividend yield = ~0.3% annual drag.
        # Scale it by 10x to make the location preference meaningful in scoring.
        tax_drag = 0.15 * 0.02 * 10  # = 0.03
        return 1.0 - tax_drag  # 0.97 multiplier

    # RRSP and Non-Reg: no US withholding drag (RRSP has treaty; Non-Reg gets FTC)
    return 1.0


def _apply_fx_and_share_rounding(
    allocations: pd.DataFrame,
    *,
    execution_config: ExecutionConfig | None,
    proxy_map: dict[str, dict[str, object]] | None,
) -> None:
    """Apply optional FX fees, compute recommended shares, and floor whole shares."""
    pre_fee_allocations = allocations["recommended_allocation"].astype(float).copy()

    if execution_config is not None and execution_config.fx_fee_bps > 0:
        fee = execution_config.fx_fee_decimal
        for idx, row in allocations.iterrows():
            ticker = row["ticker"]
            meta = get_ticker_metadata(ticker, proxy_map=proxy_map)
            if not meta["is_cad_denominated"] and row["recommended_allocation"] > 0:
                original = float(allocations.at[idx, "recommended_allocation"])  # pyright: ignore[reportArgumentType]
                allocations.at[idx, "recommended_allocation"] = original * (1.0 - fee)
                logger.debug(
                    "Applied %.2f%% FX fee to %s: $%.2f -> $%.2f",
                    fee * 100,
                    ticker,
                    original,
                    allocations.at[idx, "recommended_allocation"],
                )

    valid_prices = allocations["latest_price"].where(allocations["latest_price"] > 0)
    allocations["recommended_shares"] = (
        allocations["recommended_allocation"] / valid_prices
    )

    if execution_config is not None and not execution_config.allow_fractional:
        leftover_cash = 0.0
        for idx in allocations.index:
            raw_shares = float(allocations.at[idx, "recommended_shares"])  # pyright: ignore[reportArgumentType]
            pre_fee = float(pre_fee_allocations.at[idx])  # pyright: ignore[reportArgumentType]
            post_fee = float(allocations.at[idx, "recommended_allocation"])  # pyright: ignore[reportArgumentType]
            if pd.notna(raw_shares) and np.isfinite(raw_shares) and raw_shares > 0:
                floored = float(np.floor(raw_shares))
                price = float(allocations.at[idx, "latest_price"])  # pyright: ignore[reportArgumentType]
                allocations.at[idx, "recommended_shares"] = floored
                if floored == 0:
                    # No whole share purchased means no FX conversion or buy executes;
                    # preserve full pre-fee cash instead of consuming an FX fee.
                    allocations.at[idx, "recommended_allocation"] = 0.0
                    unspent = pre_fee
                else:
                    actual_cost = floored * price
                    allocations.at[idx, "recommended_allocation"] = actual_cost
                    unspent = post_fee - actual_cost
                leftover_cash += max(unspent, 0.0)
            elif pre_fee > 0:
                allocations.at[idx, "recommended_shares"] = 0.0
                allocations.at[idx, "recommended_allocation"] = 0.0
                leftover_cash += pre_fee
        allocations["leftover_cash"] = leftover_cash


def _zero_allocation_frame(scored: pd.DataFrame) -> pd.DataFrame:
    """Return an allocation frame with zero recommended buys for zero-cash accounts."""
    allocations = scored.copy()
    allocations["raw_allocation"] = 0.0
    allocations["recommended_allocation"] = 0.0
    allocations["spillover_allocation"] = 0.0
    allocations["spillover_account"] = None
    allocations["recalculated_score"] = np.nan
    allocations["recommended_weight_increase"] = 0.0
    allocations["allocation_rank"] = allocations["recommended_allocation"].rank(
        ascending=False, method="min"
    )
    return allocations


def _resolve_account_types(
    accounts: list[str],
    account_types: Mapping[str, str] | None,
) -> dict[str, str]:
    """Resolve explicit account->tax-type overrides, requiring full coverage."""
    if account_types is None:
        return {acct: acct for acct in accounts}

    normalised: dict[str, str] = {}
    for raw_acct, raw_type in account_types.items():
        clean_acct = str(raw_acct).strip()
        clean_type = str(raw_type).strip()
        if not clean_acct or not clean_type:
            raise DataValidationError(
                "account_types entries must have non-empty account labels and types."
            )
        if clean_acct in normalised and normalised[clean_acct] != clean_type:
            raise DataValidationError(
                f"Conflicting account_types mapping for '{clean_acct}': "
                f"{normalised[clean_acct]!r} vs {clean_type!r}."
            )
        normalised[clean_acct] = clean_type

    missing = [acct for acct in accounts if acct not in normalised]
    if missing:
        raise DataValidationError(
            "account_types must specify a tax classification for every account in the plan. "
            f"Missing: {', '.join(missing)}."
        )
    return {acct: normalised[acct] for acct in accounts}


def build_rebalance_plan(
    portfolio_name: str = "portfolio",
    *,
    new_cash: float = 0.0,
    holdings_csv: Path | None = None,
    holdings_mapping: Mapping[str, float] | None = None,
    holdings_kind: Literal["value", "shares"] | None = None,
    snapshot: HoldingsSnapshot | None = None,
    reconciliation: ReconciliationReport | None = None,
    account_by: Literal["account_type", "account_id"] = "account_type",
    deploy_existing_cash: bool = False,
    account_types: Mapping[str, str] | None = None,
    target_weights: Mapping[str, float] | Path | None = None,
    latest_prices: Mapping[str, float] | Path | None = None,
    export_dir: Path | None = None,
    config_path: Path | None = None,
    execution_config: ExecutionConfig | None = None,
    proxy_map: dict[str, dict[str, object]] | None = None,
    tax_profile: TaxProfile | None = None,
    asset_characteristics: dict[str, AssetTaxCharacteristics] | None = None,
    tax_weight: float = 0.3,
) -> RebalancePlan:
    """Create buy recommendations from optimiser outputs and live holdings.

    Parameters
    ----------
    portfolio_name : str, optional
        Portfolio identifier used to locate saved optimisation artefacts when
        ``target_weights`` or ``latest_prices`` are omitted. Defaults to
        ``"portfolio"``.
    new_cash : float, optional
        Dollar amount of new capital to allocate. Must be strictly positive
        unless ``deploy_existing_cash=True`` and existing account cash is
        positive.
    holdings_csv : pathlib.Path or None, optional
        CSV of current holdings containing a ticker column plus either
        ``current_value``/``total_value`` or ``shares``.
    holdings_mapping : Mapping[str, float] or None, optional
        In-memory alternative to ``holdings_csv`` for ticker-to-value or
        ticker-to-shares input.
    holdings_kind : {"value", "shares"} or None, optional
        Explicit interpretation of ``holdings_mapping`` values, or override for
        CSV auto-detection.
    snapshot : HoldingsSnapshot or None, optional
        Structured multi-account CAD-valued holdings snapshot from
        :func:`pysharpe.data.interop.read_holdings_snapshot`. Mutually
        exclusive with ``holdings_csv``, ``holdings_mapping``, and
        ``reconciliation``.
    reconciliation : ReconciliationReport or None, optional
        Structured ledger valuation report from
        :meth:`PortfolioLedger.reconcile`. Mutually exclusive with
        ``holdings_csv``, ``holdings_mapping``, and ``snapshot``.
    account_by : {"account_type", "account_id"}, optional
        How ``reconciliation`` groups accounts (default ``"account_type"``).
    deploy_existing_cash : bool, optional
        When ``True``, add CAD-equivalent existing cash held in ``snapshot`` or
        ``reconciliation`` accounts to ``account_cash`` alongside any
        ``new_cash``. Default is ``False``.
    account_types : Mapping[str, str] or None, optional
        Explicit mapping from each account label (e.g. ``"TFSA-1"``) to its
        canonical tax classification (e.g. ``"TFSA"``, ``"RRSP"``,
        ``"FHSA"``, ``"NON_REG"``). Required to cover all accounts when
        provided.
    target_weights : Mapping[str, float] or pathlib.Path or None, optional
        Direct ticker-to-weight mapping or CSV/TXT path, bypassing
        ``<export_dir>/<portfolio>_weights.txt``.
    latest_prices : Mapping[str, float] or pathlib.Path or None, optional
        Direct ticker-to-CAD-price mapping or CSV path. When omitted for
        ``snapshot`` / ``reconciliation`` inputs, CAD prices are derived from
        positions (and explicit ``export_dir`` collated history if needed).
    export_dir : pathlib.Path or None, optional
        Directory containing ``<portfolio>_weights.txt`` and
        ``<portfolio>_collated.csv``. Defaults to the configured export
        directory.
    config_path : pathlib.Path or None, optional
        Optional JSON config containing ``allocation_weights`` and
        ``fundamentals``.
    execution_config : ExecutionConfig or None, optional
        Execution settings controlling FX-fee application and fractional-share
        rounding.
    proxy_map : dict or None, optional
        Proxy metadata used to determine CAD denomination per ticker for
        FX-fee application.
    tax_profile : TaxProfile or None, optional
        The investor's marginal tax profile.  When provided together with
        *asset_characteristics*, the :class:`AssetLocationEngine` replaces
        the legacy heuristic tax adjustment with a mathematically rigorous
        account- and asset-specific tax drag computation.
    asset_characteristics : dict[str, AssetTaxCharacteristics] or None, optional
        Mapping of ticker to :class:`AssetTaxCharacteristics`.  Used by the
        tax-location engine when *tax_profile* is provided.
    tax_weight : float, optional
        Weight of tax efficiency in the blended opportunity score (0--1).
        Only used when the tax-location engine is active.  Default is 0.3.

    Returns
    -------
    RebalancePlan
        Fully prepared plan containing merged portfolio state, opportunity
        scores, and recommended buy orders.

    Raises
    ------
    DataValidationError
        If capital inputs, holdings sources, prices, or account mappings are
        invalid.
    DataIngestionError
        If required optimiser artefacts or config files are missing.
    """

    new_cash_val = float(new_cash)
    if not np.isfinite(new_cash_val) or new_cash_val < 0:
        raise DataValidationError(
            f"new_cash must be a positive dollar amount. Got: {new_cash}"
        )
    if not deploy_existing_cash and new_cash_val <= 0:
        raise DataValidationError(
            f"new_cash must be a positive dollar amount. Got: {new_cash}"
        )

    source_count = sum(
        src is not None
        for src in (holdings_csv, holdings_mapping, snapshot, reconciliation)
    )
    if source_count == 0:
        raise DataValidationError(
            "No holdings provided. Specify holdings_csv, holdings_mapping, "
            "snapshot, or reconciliation."
        )
    if source_count > 1:
        raise DataValidationError(
            "Provide only one holdings input among holdings_csv, "
            "holdings_mapping, snapshot, and reconciliation."
        )

    settings = get_settings()
    export_dir_explicit = export_dir is not None
    export_root = Path(export_dir or settings.export_dir).expanduser().resolve()

    resolved_weights, weights_path = _resolve_target_weights(
        portfolio_name, export_root, target_weights
    )
    target_tickers = set(resolved_weights["ticker"].astype(str).tolist())
    resolved_prices, prices_path = _resolve_latest_prices(
        portfolio_name,
        export_root,
        latest_prices,
        export_dir_explicit=export_dir_explicit,
        snapshot=snapshot,
        reconciliation=reconciliation,
        target_tickers=target_tickers,
    )

    existing_cash_by_account: dict[str, float] = {}
    if snapshot is not None:
        holdings, existing_cash_by_account = _holdings_from_snapshot(snapshot)
    elif reconciliation is not None:
        holdings, existing_cash_by_account = _holdings_from_reconciliation(
            reconciliation, account_by=account_by
        )
    else:
        holdings = _load_holdings_frame(
            holdings_csv=holdings_csv,
            holdings_mapping=holdings_mapping,
            holdings_kind=holdings_kind,
            latest_prices=resolved_prices,
        )

    allocation_config, fundamentals = _load_allocation_inputs(config_path)

    # --- Detect multi-account mode ---
    has_accounts = (
        snapshot is not None
        or reconciliation is not None
        or "account" in holdings.columns
    )

    if not has_accounts:
        if deploy_existing_cash and new_cash_val <= 0:
            raise DataValidationError(
                "Total deployable cash must be positive when deploy_existing_cash=True."
            )
        # === Single-account path (existing behavior) ===
        merged = resolved_weights.merge(resolved_prices, on="ticker", how="outer")
        merged = merged.merge(holdings, on="ticker", how="outer")
        merged = _merge_fundamentals(merged, fundamentals)

        merged["target_weight"] = pd.to_numeric(
            merged["target_weight"], errors="coerce"
        ).fillna(0.0)
        merged["current_value"] = pd.to_numeric(
            merged["current_value"], errors="coerce"
        ).fillna(0.0)
        if "current_shares" in merged.columns:
            merged["current_shares"] = pd.to_numeric(
                merged["current_shares"], errors="coerce"
            ).fillna(0.0)

        scored = score_opportunities(merged, config=allocation_config)
        allocations = allocate_contribution(
            scored,
            contribution_dollars=new_cash_val,
            config=allocation_config,
        )

        # --- FX fee application & whole-share rounding ---
        _apply_fx_and_share_rounding(
            allocations,
            execution_config=execution_config,
            proxy_map=proxy_map,
        )

        total_after = allocations["current_value"].sum() + new_cash_val
        if total_after > 0:
            allocations["estimated_post_buy_weight"] = (
                allocations["current_value"] + allocations["recommended_allocation"]
            ) / total_after
        else:
            allocations["estimated_post_buy_weight"] = 0.0

        return RebalancePlan(
            portfolio_name=portfolio_name,
            new_cash=new_cash_val,
            weights_path=weights_path,
            prices_path=prices_path,
            scored_state=scored,
            allocations=allocations,
        )

    # === Multi-account path ===
    position_accounts = (
        set(holdings["account"].unique()) if not holdings.empty else set()
    )
    cash_accounts = (
        {acct for acct, cash in existing_cash_by_account.items() if cash > 0}
        if deploy_existing_cash
        else set()
    )
    accounts = sorted(position_accounts | cash_accounts)
    if not accounts:
        raise DataValidationError(
            "Holdings contain no accounts with positions or deployable cash."
        )

    resolved_types = _resolve_account_types(accounts, account_types)
    account_values = (
        holdings.groupby("account")["current_value"].sum()
        if not holdings.empty
        else pd.Series(dtype=float)
    )
    total_value = float(account_values.sum())

    # Split new cash proportionally by current account value
    if new_cash_val == 0:
        split_new_cash = {acct: 0.0 for acct in accounts}
    elif total_value > 0:
        split_new_cash = {
            acct: round(new_cash_val * (account_values.get(acct, 0.0) / total_value), 2)
            for acct in accounts
        }
        # Distribute rounding remainder to the largest account
        remainder = round(new_cash_val - sum(split_new_cash.values()), 2)
        if remainder != 0 and accounts:
            largest = max(accounts, key=lambda a: account_values.get(a, 0.0))
            split_new_cash[largest] = round(split_new_cash[largest] + remainder, 2)
    else:
        split = round(new_cash_val / len(accounts), 2)
        split_new_cash = {acct: split for acct in accounts}
        remainder = round(new_cash_val - sum(split_new_cash.values()), 2)
        if remainder != 0 and accounts:
            split_new_cash[accounts[0]] = round(
                split_new_cash[accounts[0]] + remainder, 2
            )

    if deploy_existing_cash:
        account_cash = {
            acct: round(
                split_new_cash[acct] + existing_cash_by_account.get(acct, 0.0), 2
            )
            for acct in accounts
        }
        effective_new_cash = round(sum(account_cash.values()), 2)
        if effective_new_cash <= 0:
            raise DataValidationError(
                "Total deployable cash (new_cash + existing cash) must be positive when deploy_existing_cash=True."
            )
    else:
        account_cash = split_new_cash
        effective_new_cash = new_cash_val

    all_scored: list[pd.DataFrame] = []
    account_allocations: dict[str, pd.DataFrame] = {}

    for account in accounts:
        tax_account = resolved_types[account]
        acct_holdings = holdings[holdings["account"] == account].drop(
            columns=["account"]
        )

        merged = resolved_weights.merge(resolved_prices, on="ticker", how="outer")
        merged = merged.merge(acct_holdings, on="ticker", how="outer")
        merged = _merge_fundamentals(merged, fundamentals)

        merged["target_weight"] = pd.to_numeric(
            merged["target_weight"], errors="coerce"
        ).fillna(0.0)
        merged["current_value"] = pd.to_numeric(
            merged["current_value"], errors="coerce"
        ).fillna(0.0)
        if "current_shares" in merged.columns:
            merged["current_shares"] = pd.to_numeric(
                merged["current_shares"], errors="coerce"
            ).fillna(0.0)

        scored = score_opportunities(merged, config=allocation_config)

        # --- Tax-location adjustment ---
        if tax_profile is not None and asset_characteristics is not None:
            engine = AssetLocationEngine(tax_profile)
            for idx in scored.index:
                ticker = str(scored.at[idx, "ticker"])
                asset = asset_characteristics.get(ticker)
                if asset is not None:
                    tax_eff = engine.compute_tax_efficiency_score(asset, tax_account)
                    base = float(scored.at[idx, "opportunity_score"])  # pyright: ignore[reportArgumentType]
                    scored.at[idx, "opportunity_score"] = (
                        1.0 - tax_weight
                    ) * base + tax_weight * tax_eff
                    drag = engine.compute_total_drag(asset, tax_account)
                    logger.debug(
                        "Tax-location for %s in %s (%s): drag=%.4f%% "
                        "tax_eff=%.4f score %.4f->%.4f",
                        ticker,
                        account,
                        tax_account,
                        drag * 100,
                        tax_eff,
                        base,
                        scored.at[idx, "opportunity_score"],
                    )
        else:
            for idx in scored.index:
                ticker = str(scored.at[idx, "ticker"])
                adjustment = _determine_account_tax_adjustment(
                    tax_account, ticker, proxy_map=proxy_map
                )
                if adjustment < 1.0:
                    current_score = float(scored.at[idx, "opportunity_score"])  # pyright: ignore[reportArgumentType]
                    scored.at[idx, "opportunity_score"] = current_score * adjustment
                    logger.debug(
                        "Tax-location penalty x%.2f for %s in %s (%s)",
                        adjustment,
                        ticker,
                        account,
                        tax_account,
                    )

        if account_cash[account] > 0:
            allocations = allocate_contribution(
                scored,
                contribution_dollars=account_cash[account],
                config=allocation_config,
            )
        else:
            allocations = _zero_allocation_frame(scored)

        # --- FX fee application & whole-share rounding ---
        _apply_fx_and_share_rounding(
            allocations,
            execution_config=execution_config,
            proxy_map=proxy_map,
        )

        total_after = allocations["current_value"].sum() + account_cash[account]
        if total_after > 0:
            allocations["estimated_post_buy_weight"] = (
                allocations["current_value"] + allocations["recommended_allocation"]
            ) / total_after
        else:
            allocations["estimated_post_buy_weight"] = 0.0

        # Tag allocations with account for downstream grouping
        allocations["account"] = account
        account_allocations[account] = allocations
        all_scored.append(scored.assign(account=account))

    # Build combined view for backward compatibility
    combined_allocations = pd.concat(account_allocations.values(), ignore_index=True)
    combined_scored = pd.concat(all_scored, ignore_index=True)

    return RebalancePlan(
        portfolio_name=portfolio_name,
        new_cash=effective_new_cash,
        weights_path=weights_path,
        prices_path=prices_path,
        scored_state=combined_scored,
        allocations=combined_allocations,
        account_allocations=account_allocations,
        account_cash=account_cash,
    )


def _format_money(value: float | int | None) -> str:
    """Format a number as a dollar amount for terminal output."""

    if value is None or pd.isna(value):
        return "n/a"
    return f"${float(value):,.2f}"


def _format_percent(value: float | int | None) -> str:
    """Format a decimal weight as a percentage string."""

    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def _format_shares(value: float | int | None) -> str:
    """Format a share quantity for terminal output."""

    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.4f}"


def format_rebalance_plan(
    plan: RebalancePlan,
    *,
    include_zero_buys: bool = False,
) -> str:
    """Render a human-readable rebalance table for terminal output.

    When the plan contains per-account allocations (multi-account mode),
    buy orders are grouped under account headers (e.g. ``TFSA Buy Orders``).

    Parameters
    ----------
    plan : RebalancePlan
        Generated rebalance plan to render.
    include_zero_buys : bool, optional
        When ``True``, include tickers that did not receive any recommended
        allocation.

    Returns
    -------
    str
        Multi-line string containing the portfolio name, new cash amount, and
        per-ticker buy recommendations in dollars and shares.
    """
    lines = [
        f"Rebalance plan for {plan.portfolio_name}",
        f"New cash: {_format_money(plan.new_cash)}",
    ]

    if plan.is_multi_account:
        assert plan.account_allocations is not None
        assert plan.account_cash is not None

        for account in plan.accounts:
            acct_alloc = plan.account_allocations[account].copy()
            if not include_zero_buys:
                acct_alloc = acct_alloc.loc[
                    acct_alloc["recommended_allocation"] > 0
                ].copy()

            acct_cash = plan.account_cash.get(account, 0.0)
            lines.append("")
            lines.append(f"\u2500\u2500 {account} (${acct_cash:,.2f}) \u2500\u2500")

            if acct_alloc.empty:
                lines.append("  No buys recommended for this account.")
                continue

            leftover = (
                acct_alloc["leftover_cash"].iloc[0]
                if "leftover_cash" in acct_alloc.columns
                else 0.0
            )
            if leftover and leftover > 0:
                lines.append(
                    f"  Unallocated (fractional leftover): {_format_money(leftover)}"
                )

            display = pd.DataFrame(
                {
                    "Ticker": acct_alloc["ticker"],
                    "Price": acct_alloc["latest_price"].map(_format_money),
                    "Target": acct_alloc["target_weight"].map(_format_percent),
                    "Current": acct_alloc["current_weight"].map(_format_percent),
                    "Score": acct_alloc["opportunity_score"].map(lambda v: f"{v:.4f}"),
                    "Buy $": acct_alloc["recommended_allocation"].map(_format_money),
                    "Buy Shares": acct_alloc["recommended_shares"].map(_format_shares),
                }
            )
            lines.append(display.to_string(index=False))
    else:
        rows = plan.allocations.copy()
        if not include_zero_buys:
            rows = rows.loc[rows["recommended_allocation"] > 0].copy()

        if rows.empty:
            lines.extend(["", "No buys recommended."])
        else:
            leftover = (
                plan.allocations["leftover_cash"].iloc[0]
                if "leftover_cash" in plan.allocations.columns
                else 0.0
            )
            if leftover and leftover > 0:
                lines.append(
                    f"Unallocated (fractional leftover): {_format_money(leftover)}"
                )

            display = pd.DataFrame(
                {
                    "Ticker": rows["ticker"],
                    "Price": rows["latest_price"].map(_format_money),
                    "Target": rows["target_weight"].map(_format_percent),
                    "Current": rows["current_weight"].map(_format_percent),
                    "Score": rows["opportunity_score"].map(lambda v: f"{v:.4f}"),
                    "Buy $": rows["recommended_allocation"].map(_format_money),
                    "Buy Shares": rows["recommended_shares"].map(_format_shares),
                }
            )
            lines.append("")
            lines.append(display.to_string(index=False))

    return "\n".join(lines)


def _parse_fx_flags(values: Sequence[str] | None) -> dict[str, float]:
    """Parse repeatable ``--fx CCY=RATE`` CLI flags into a rate mapping."""
    if not values:
        return {}
    rates: dict[str, float] = {}
    for entry in values:
        if "=" not in entry:
            raise DataValidationError(
                f"Invalid --fx specification {entry!r}. Expected CURRENCY=RATE (e.g. --fx USD=1.35)."
            )
        raw_ccy, raw_rate = entry.split("=", 1)
        ccy = raw_ccy.strip().upper()
        if not ccy or not ccy.isalpha():
            raise DataValidationError(
                f"Invalid currency code in --fx {entry!r}. Expected alphabetic code such as USD."
            )
        try:
            rate = float(raw_rate.strip())
        except ValueError as exc:
            raise DataValidationError(
                f"FX rate for {ccy} must be numeric; got {raw_rate!r}."
            ) from exc
        if not np.isfinite(rate) or rate <= 0:
            raise DataValidationError(
                f"FX rate for {ccy} must be finite and positive; got {raw_rate!r}."
            )
        existing = rates.get(ccy)
        if existing is not None and not np.isclose(
            existing, rate, rtol=1e-9, atol=1e-9
        ):
            raise DataValidationError(
                f"Conflicting --fx rates supplied for {ccy}: {existing} vs {rate}."
            )
        rates[ccy] = rate
    return rates


def _parse_as_of(raw: str | None) -> date:
    """Parse an optional ``YYYY-MM-DD`` snapshot date, defaulting to today."""
    if raw is None:
        return date.today()
    try:
        return date.fromisoformat(raw.strip())
    except ValueError as exc:
        raise DataValidationError(
            f"Invalid --as-of date {raw!r}. Expected ISO format YYYY-MM-DD."
        ) from exc


def _resolve_cli_mapping_or_path(
    raw: str,
    *,
    label: str,
    allow_zero: bool,
) -> dict[str, float] | Path:
    """Resolve CLI ``--weights`` / ``--prices`` value as JSON mapping or CSV path."""
    candidate = Path(raw).expanduser()
    try:
        is_file = candidate.exists()
    except OSError:
        is_file = False

    if is_file:
        if candidate.suffix.lower() in {".csv", ".txt"}:
            return candidate
        return _parse_numeric_json_mapping(raw, label=label, allow_zero=allow_zero)

    if raw.lstrip().startswith("{"):
        return _parse_numeric_json_mapping(raw, label=label, allow_zero=allow_zero)

    return candidate


def add_rebalance_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_allow_fractional: bool = False,
) -> None:
    """Register shared rebalance CLI options on *parser*."""
    parser.add_argument(
        "--portfolio",
        default="portfolio",
        help=(
            "Portfolio name whose <portfolio>_weights.txt and "
            "<portfolio>_collated.csv are used unless --weights/--prices "
            "are given (default: 'portfolio')."
        ),
    )
    holdings = parser.add_mutually_exclusive_group(required=True)
    holdings.add_argument(
        "--holdings-csv",
        type=Path,
        help=(
            "CSV containing a ticker column and either "
            "current_value/total_value or shares."
        ),
    )
    holdings.add_argument(
        "--holdings-json",
        help=(
            "Inline JSON object or path to a JSON file mapping "
            "ticker to value/share count."
        ),
    )
    holdings.add_argument(
        "--snapshot-csv",
        type=Path,
        help=(
            "Multi-currency brokerage holdings CSV imported via "
            "read_holdings_snapshot (supports --fx and --normalise-tickers)."
        ),
    )
    parser.add_argument(
        "--holdings-kind",
        choices=("value", "shares"),
        help="Interpret holdings JSON values, or override CSV auto-detection.",
    )
    parser.add_argument(
        "--new-cash",
        type=float,
        default=None,
        help=(
            "Dollar amount of new capital to allocate. Required unless "
            "--deploy-existing-cash is specified."
        ),
    )
    parser.add_argument(
        "--deploy-existing-cash",
        action="store_true",
        help=(
            "Include CAD-equivalent account cash balances (from --snapshot-csv "
            "or --cash) in deployable allocation budgets."
        ),
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Snapshot valuation date (YYYY-MM-DD) used with --snapshot-csv.",
    )
    parser.add_argument(
        "--fx",
        action="append",
        dest="fx_rates",
        metavar="CURRENCY=RATE",
        default=None,
        help=(
            "CAD exchange rate for non-CAD holdings or cash in --snapshot-csv, "
            "e.g. --fx USD=1.35. Repeatable."
        ),
    )
    parser.add_argument(
        "--normalise-tickers",
        action="store_true",
        help=(
            "Strip exchange suffixes (.TSE, .NYSE, .TO, etc.) from brokerage "
            "tickers when reading --snapshot-csv."
        ),
    )
    parser.add_argument(
        "--cash",
        default=None,
        help=(
            "Inline JSON or JSON file mapping account -> amount or "
            "{currency: amount} for --snapshot-csv."
        ),
    )
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Direct target weights as inline JSON, JSON file, or CSV/TXT path, "
            "bypassing <export_dir>/<portfolio>_weights.txt."
        ),
    )
    parser.add_argument(
        "--prices",
        default=None,
        help=(
            "Direct latest CAD prices as inline JSON, JSON file, or CSV path. "
            "With --snapshot-csv, defaults to position prices when sufficient."
        ),
    )
    parser.add_argument(
        "--account-types",
        default=None,
        help=(
            "Inline JSON or JSON file mapping account label (e.g. TFSA-1) to "
            "canonical tax type (TFSA, RRSP, FHSA, NON_REG)."
        ),
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help=(
            "Directory containing optimisation artefacts. "
            "Defaults to the configured export directory."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config with allocation_weights and fundamentals.",
    )
    parser.add_argument(
        "--include-zero-buys",
        action="store_true",
        help=(
            "Show the full merged portfolio state instead of only "
            "positive buy recommendations."
        ),
    )
    if include_allow_fractional:
        parser.add_argument(
            "--allow-fractional",
            action="store_true",
            default=None,
            help=(
                "Allow fractional share purchases. When set, overrides "
                "the portfolio_config.json setting to True."
            ),
        )


def run_rebalance_cli(
    args: argparse.Namespace,
    *,
    execution_config: ExecutionConfig | None = None,
    proxy_map: dict[str, dict[str, object]] | None = None,
) -> int:
    """Execute the rebalance workflow for parsed CLI arguments."""
    from pysharpe.data.interop import normalise_ticker, read_holdings_snapshot

    deploy_existing_cash = bool(getattr(args, "deploy_existing_cash", False))
    raw_new_cash = getattr(args, "new_cash", None)
    if raw_new_cash is None:
        if deploy_existing_cash:
            new_cash = 0.0
        else:
            print("Error: --new-cash is required unless --deploy-existing-cash is set.")
            return 1
    else:
        new_cash = float(raw_new_cash)

    holdings_mapping = None
    snapshot = None
    target_weights: Mapping[str, float] | Path | None = None
    latest_prices: Mapping[str, float] | Path | None = None
    account_types: Mapping[str, str] | None = None

    try:
        if args.holdings_json:
            holdings_mapping = _load_holdings_mapping(args.holdings_json)
        if getattr(args, "snapshot_csv", None) is not None:
            fx_rates = _parse_fx_flags(getattr(args, "fx_rates", None))
            as_of = _parse_as_of(getattr(args, "as_of", None))
            cash = (
                _parse_cash_json(args.cash)
                if getattr(args, "cash", None) is not None
                else None
            )
            ticker_transform = (
                normalise_ticker if getattr(args, "normalise_tickers", False) else None
            )
            snapshot = read_holdings_snapshot(
                args.snapshot_csv,
                ticker_transform=ticker_transform,
                as_of=as_of,
                fx_rates=fx_rates,
                cash=cash,
            )
        elif getattr(args, "fx_rates", None) or getattr(args, "cash", None):
            raise DataValidationError("--fx and --cash require --snapshot-csv.")

        if getattr(args, "weights", None) is not None:
            target_weights = _resolve_cli_mapping_or_path(
                args.weights, label="Target weights", allow_zero=True
            )
        if getattr(args, "prices", None) is not None:
            latest_prices = _resolve_cli_mapping_or_path(
                args.prices, label="Latest prices", allow_zero=False
            )
        if getattr(args, "account_types", None) is not None:
            account_types = _parse_account_types_json(args.account_types)

        plan = build_rebalance_plan(
            args.portfolio,
            new_cash=new_cash,
            holdings_csv=args.holdings_csv,
            holdings_mapping=holdings_mapping,
            holdings_kind=args.holdings_kind,
            snapshot=snapshot,
            deploy_existing_cash=deploy_existing_cash,
            account_types=account_types,
            target_weights=target_weights,
            latest_prices=latest_prices,
            export_dir=args.export_dir,
            config_path=args.config,
            execution_config=execution_config,
            proxy_map=proxy_map,
        )
    except (
        FileNotFoundError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        PySharpeError,
    ) as exc:
        print(f"Error: {exc}")
        return 1

    print(format_rebalance_plan(plan, include_zero_buys=args.include_zero_buys))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Construct the standalone rebalance CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="rebalance.py",
        description="Build buy orders from saved PySharpe optimisation outputs or snapshots.",
    )
    add_rebalance_arguments(parser, include_allow_fractional=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone rebalance command-line interface."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    return run_rebalance_cli(args)


__all__ = [
    "RebalancePlan",
    "add_rebalance_arguments",
    "build_rebalance_plan",
    "format_rebalance_plan",
    "main",
    "run_rebalance_cli",
]
