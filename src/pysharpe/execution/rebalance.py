"""Build contribution plans from saved optimisation artefacts."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    weights_path : Path
        Path to the saved optimiser output containing target weights.
    prices_path : Path
        Path to the collated history used to derive the latest prices.
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
    weights_path: Path
    prices_path: Path
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


def _load_target_weights(
    portfolio_name: str,
    export_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    """Load target weights emitted by the optimisation workflow.

    Parameters
    ----------
    portfolio_name : str
        Portfolio identifier used to resolve ``<portfolio>_weights.txt``.
    export_dir : pathlib.Path
        Directory containing optimiser artefacts.

    Returns
    -------
    tuple[pandas.DataFrame, pathlib.Path]
        Two-item tuple containing the normalized target-weight table and the
        path it was loaded from.


    Raises
    ------
    DataIngestionError
        If the expected weights file is missing, empty, or unparseable.
    DataValidationError
        If the file does not contain the required ticker/weight columns.
    """

    path = export_dir / f"{portfolio_name}_weights.txt"
    if not path.exists():
        raise DataIngestionError(
            f"Optimisation weights file not found for portfolio "
            f"'{portfolio_name}' at: {path}\n"
            f"Run 'pysharpe optimise --portfolio {portfolio_name}' first to "
            f"generate the weights."
        )

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


def _load_latest_prices(
    portfolio_name: str,
    export_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    """Extract the most recent non-null prices from a collated history file.

    Parameters
    ----------
    portfolio_name : str
        Portfolio identifier used to resolve ``<portfolio>_collated.csv``.
    export_dir : pathlib.Path
        Directory containing collated price histories.

    Returns
    -------
    tuple[pandas.DataFrame, pathlib.Path]
        Latest price table with ``ticker`` and ``latest_price`` columns, plus
        the source CSV path.

    Raises
    ------
    DataIngestionError
        If the collated history file is missing, empty, or unparseable.
    """

    path = export_dir / f"{portfolio_name}_collated.csv"
    if not path.exists():
        raise DataIngestionError(
            f"Collated price history not found for portfolio "
            f"'{portfolio_name}' at: {path}\n"
            f"Run 'pysharpe optimise --portfolio {portfolio_name}' first to "
            f"generate it."
        )

    try:
        frame = pd.read_csv(path, index_col=0)
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


def _load_holdings_mapping(raw: str) -> dict[str, float]:
    """Parse inline JSON or a JSON file path into a holdings mapping.

    Parameters
    ----------
    raw : str
        Either a JSON object string or a path to a JSON file.

    Returns
    -------
    dict[str, float]
        Mapping of ticker symbol to current value or share count.

    Raises
    ------
    DataValidationError
        If the payload is not a valid non-empty ticker-to-number mapping.
    DataIngestionError
        If the holdings JSON file cannot be read.
    """

    candidate = Path(raw).expanduser()
    if candidate.exists():
        try:
            payload = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise DataIngestionError(
                f"Cannot read holdings file: {candidate}\nDetails: {exc}"
            ) from exc
    else:
        payload = raw

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DataValidationError(
            f"Holdings JSON is not valid. Check for missing commas, "
            f"trailing commas, or unquoted keys.\n"
            f"Details: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise DataValidationError(
            "Holdings JSON must be an object mapping ticker to value."
        )

    parsed: dict[str, float] = {}
    for ticker, value in data.items():
        clean_ticker = str(ticker).strip()
        if not clean_ticker:
            continue

        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            raise DataValidationError(
                f"Invalid holdings value for ticker '{clean_ticker}'. "
                f"Value must be a number, got: {value!r}"
            )
        if numeric < 0:
            raise DataValidationError(
                f"Holdings value cannot be negative for ticker "
                f"'{clean_ticker}'. Got: {numeric}"
            )
        parsed[clean_ticker] = float(numeric)

    if not parsed:
        raise DataValidationError(
            "Holdings JSON did not contain any valid ticker entries. "
            "Provide at least one ticker with a numeric value."
        )
    return parsed


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


def build_rebalance_plan(
    portfolio_name: str,
    *,
    new_cash: float,
    holdings_csv: Path | None = None,
    holdings_mapping: Mapping[str, float] | None = None,
    holdings_kind: Literal["value", "shares"] | None = None,
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
    portfolio_name : str
        Portfolio identifier used to locate saved optimisation artefacts.
    new_cash : float
        Dollar amount of new capital to allocate.
    holdings_csv : pathlib.Path or None, optional
        CSV of current holdings containing a ticker column plus either
        ``current_value``/``total_value`` or ``shares``.
    holdings_mapping : Mapping[str, float] or None, optional
        In-memory alternative to ``holdings_csv`` for ticker-to-value or
        ticker-to-shares input.
    holdings_kind : {"value", "shares"} or None, optional
        Explicit interpretation of ``holdings_mapping`` values, or override for
        CSV auto-detection.
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
        If ``new_cash`` is not positive or the holdings input is invalid.
    DataIngestionError
        If required optimiser artefacts or config files are missing.
    """

    if new_cash <= 0:
        raise DataValidationError(
            f"new_cash must be a positive dollar amount. Got: {new_cash}"
        )

    settings = get_settings()
    export_root = Path(export_dir or settings.export_dir).expanduser().resolve()

    target_weights, weights_path = _load_target_weights(portfolio_name, export_root)
    latest_prices, prices_path = _load_latest_prices(portfolio_name, export_root)
    holdings = _load_holdings_frame(
        holdings_csv=holdings_csv,
        holdings_mapping=holdings_mapping,
        holdings_kind=holdings_kind,
        latest_prices=latest_prices,
    )
    allocation_config, fundamentals = _load_allocation_inputs(config_path)

    # --- Detect multi-account mode ---
    has_accounts = "account" in holdings.columns

    if not has_accounts:
        # === Single-account path (existing behavior) ===
        merged = target_weights.merge(latest_prices, on="ticker", how="outer")
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
            contribution_dollars=new_cash,
            config=allocation_config,
        )

        # --- FX fee application ---
        if execution_config is not None and execution_config.fx_fee_bps > 0:
            for idx, row in allocations.iterrows():
                ticker = row["ticker"]
                meta = get_ticker_metadata(ticker, proxy_map=proxy_map)
                if not meta["is_cad_denominated"] and row["recommended_allocation"] > 0:
                    original = row["recommended_allocation"]
                    fee = execution_config.fx_fee_decimal
                    allocations.at[idx, "recommended_allocation"] *= 1.0 - fee
                    logger.debug(
                        "Applied %.2f%% FX fee to %s: $%.2f -> $%.2f",
                        fee * 100,
                        ticker,
                        original,
                        allocations.at[idx, "recommended_allocation"],
                    )

        valid_prices = allocations["latest_price"].where(
            allocations["latest_price"] > 0
        )
        allocations["recommended_shares"] = (
            allocations["recommended_allocation"] / valid_prices
        )

        leftover_cash = 0.0
        if execution_config is not None and not execution_config.allow_fractional:
            for idx in allocations.index:
                raw_shares = allocations.at[idx, "recommended_shares"]
                if pd.notna(raw_shares) and raw_shares > 0:
                    floored = np.floor(raw_shares)
                    price = allocations.at[idx, "latest_price"]
                    allocations.at[idx, "recommended_shares"] = floored
                    actual_cost = floored * price
                    unspent = (
                        allocations.at[idx, "recommended_allocation"] - actual_cost
                    )
                    allocations.at[idx, "recommended_allocation"] = actual_cost
                    leftover_cash += max(unspent, 0.0)
            allocations["leftover_cash"] = leftover_cash

        total_after = allocations["current_value"].sum() + new_cash
        if total_after > 0:
            allocations["estimated_post_buy_weight"] = (
                allocations["current_value"] + allocations["recommended_allocation"]
            ) / total_after
        else:
            allocations["estimated_post_buy_weight"] = 0.0

        return RebalancePlan(
            portfolio_name=portfolio_name,
            new_cash=new_cash,
            weights_path=weights_path,
            prices_path=prices_path,
            scored_state=scored,
            allocations=allocations,
        )

    # === Multi-account path ===
    accounts = sorted(holdings["account"].unique())
    account_values = holdings.groupby("account")["current_value"].sum()
    total_value = account_values.sum()

    # Split new cash proportionally by current account value
    if total_value > 0:
        account_cash = {
            acct: round(new_cash * (account_values.get(acct, 0.0) / total_value), 2)
            for acct in accounts
        }
        # Distribute rounding remainder to the largest account
        remainder = new_cash - sum(account_cash.values())
        if remainder != 0 and accounts:
            largest = max(accounts, key=lambda a: account_values.get(a, 0.0))
            account_cash[largest] = round(account_cash[largest] + remainder, 2)
    else:
        split = new_cash / len(accounts)
        account_cash = {acct: split for acct in accounts}

    all_scored: list[pd.DataFrame] = []
    account_allocations: dict[str, pd.DataFrame] = {}

    for account in accounts:
        acct_holdings = holdings[holdings["account"] == account].drop(
            columns=["account"]
        )

        merged = target_weights.merge(latest_prices, on="ticker", how="outer")
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
                    tax_eff = engine.compute_tax_efficiency_score(asset, account)
                    base = scored.at[idx, "opportunity_score"]
                    scored.at[idx, "opportunity_score"] = (
                        1.0 - tax_weight
                    ) * base + tax_weight * tax_eff
                    drag = engine.compute_total_drag(asset, account)
                    logger.debug(
                        "Tax-location for %s in %s: drag=%.4f%% "
                        "tax_eff=%.4f score %.4f->%.4f",
                        ticker,
                        account,
                        drag * 100,
                        tax_eff,
                        base,
                        scored.at[idx, "opportunity_score"],
                    )
        else:
            for idx in scored.index:
                ticker = str(scored.at[idx, "ticker"])
                adjustment = _determine_account_tax_adjustment(
                    account, ticker, proxy_map=proxy_map
                )
                if adjustment < 1.0:
                    scored.at[idx, "opportunity_score"] *= adjustment
                    logger.debug(
                        "Tax-location penalty x%.2f for %s in %s",
                        adjustment,
                        ticker,
                        account,
                    )

        allocations = allocate_contribution(
            scored,
            contribution_dollars=account_cash[account],
            config=allocation_config,
        )

        # --- FX fee application ---
        if execution_config is not None and execution_config.fx_fee_bps > 0:
            for idx, row in allocations.iterrows():
                ticker = row["ticker"]
                meta = get_ticker_metadata(ticker, proxy_map=proxy_map)
                if not meta["is_cad_denominated"] and row["recommended_allocation"] > 0:
                    fee = execution_config.fx_fee_decimal
                    allocations.at[idx, "recommended_allocation"] *= 1.0 - fee

        valid_prices = allocations["latest_price"].where(
            allocations["latest_price"] > 0
        )
        allocations["recommended_shares"] = (
            allocations["recommended_allocation"] / valid_prices
        )

        leftover_cash = 0.0
        if execution_config is not None and not execution_config.allow_fractional:
            for idx in allocations.index:
                raw_shares = allocations.at[idx, "recommended_shares"]
                if pd.notna(raw_shares) and raw_shares > 0:
                    floored = np.floor(raw_shares)
                    price = allocations.at[idx, "latest_price"]
                    allocations.at[idx, "recommended_shares"] = floored
                    actual_cost = floored * price
                    unspent = (
                        allocations.at[idx, "recommended_allocation"] - actual_cost
                    )
                    allocations.at[idx, "recommended_allocation"] = actual_cost
                    leftover_cash += max(unspent, 0.0)
            allocations["leftover_cash"] = leftover_cash

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
        new_cash=new_cash,
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


def _build_parser() -> argparse.ArgumentParser:
    """Construct the standalone rebalance CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for the standalone ``scripts/rebalance.py`` entry
        point.
    """

    parser = argparse.ArgumentParser(
        prog="rebalance.py",
        description="Build buy orders from saved PySharpe optimisation outputs.",
    )
    parser.add_argument(
        "--portfolio",
        required=True,
        help=(
            "Portfolio name whose <portfolio>_weights.txt and "
            "<portfolio>_collated.csv will be used."
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
    parser.add_argument(
        "--holdings-kind",
        choices=("value", "shares"),
        help="Interpret holdings JSON values, or override CSV auto-detection.",
    )
    parser.add_argument(
        "--new-cash",
        type=float,
        required=True,
        help="Dollar amount of new capital to allocate.",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone rebalance command-line interface.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Explicit argument vector. When ``None``, arguments are read from the
        process command line.

    Returns
    -------
    int
        Process exit code. Returns ``0`` on success and ``1`` on user-facing
        validation or file errors.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    holdings_mapping = None
    if args.holdings_json:
        try:
            holdings_mapping = _load_holdings_mapping(args.holdings_json)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"Error: {exc}")
            return 1

    try:
        plan = build_rebalance_plan(
            args.portfolio,
            new_cash=args.new_cash,
            holdings_csv=args.holdings_csv,
            holdings_mapping=holdings_mapping,
            holdings_kind=args.holdings_kind,
            export_dir=args.export_dir,
            config_path=args.config,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(format_rebalance_plan(plan, include_zero_buys=args.include_zero_buys))
    return 0


__all__ = ["RebalancePlan", "build_rebalance_plan", "format_rebalance_plan", "main"]
