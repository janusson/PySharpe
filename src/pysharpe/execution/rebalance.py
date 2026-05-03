"""Build contribution plans from saved optimisation artefacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import pandas as pd

from pysharpe.config import get_settings

from .allocator import AllocationConfig, allocate_contribution, score_opportunities

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
    """

    portfolio_name: str
    new_cash: float
    weights_path: Path
    prices_path: Path
    scored_state: pd.DataFrame
    allocations: pd.DataFrame

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
        raise ValueError(f"Invalid {label} values for tickers: {', '.join(invalid)}")

    negative = frame.loc[values < 0, "ticker"].tolist()
    if negative:
        raise ValueError(f"{label.capitalize()} cannot be negative: {', '.join(negative)}")

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
        raise ValueError("Input data must include a ticker column.")

    prepared = frame.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.strip()
    prepared = prepared.loc[prepared["ticker"] != ""].copy()
    if prepared.empty:
        raise ValueError("Holdings data is empty after removing blank tickers.")
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
    FileNotFoundError
        If the expected weights file is missing.
    ValueError
        If the file does not contain the required ticker/weight columns.
    """

    path = export_dir / f"{portfolio_name}_weights.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Optimisation weights not found for {portfolio_name}: {path}"
        )

    frame = _rename_known_columns(pd.read_csv(path))
    if "ticker" not in frame.columns:
        raise ValueError(f"Weights file is missing a ticker column: {path}")
    if "weight" not in frame.columns and "target_weight" not in frame.columns:
        raise ValueError(f"Weights file is missing a weight column: {path}")

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
    FileNotFoundError
        If the collated history file is missing.
    ValueError
        If prices cannot be derived from the file contents.
    """

    path = export_dir / f"{portfolio_name}_collated.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Collated price history not found for {portfolio_name}: {path}"
        )

    frame = pd.read_csv(path, index_col=0)
    if frame.empty:
        raise ValueError(f"Collated price history is empty: {path}")

    prices = frame.apply(pd.to_numeric, errors="coerce").ffill()
    if prices.empty:
        raise ValueError(
            f"Collated price history has no numeric price columns: {path}"
        )

    latest = prices.iloc[-1].dropna()
    if latest.empty:
        raise ValueError(f"Unable to derive latest prices from {path}")

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
    ValueError
        If the payload is not a valid non-empty ticker-to-number mapping.
    """

    candidate = Path(raw).expanduser()
    payload = candidate.read_text(encoding="utf-8") if candidate.exists() else raw
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Holdings JSON must be an object mapping ticker to value.")

    parsed: dict[str, float] = {}
    for ticker, value in data.items():
        clean_ticker = str(ticker).strip()
        if not clean_ticker:
            continue

        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            raise ValueError(f"Invalid holdings value for ticker {clean_ticker}.")
        if numeric < 0:
            raise ValueError(
                f"Holdings value cannot be negative for ticker {clean_ticker}."
            )
        parsed[clean_ticker] = float(numeric)

    if not parsed:
        raise ValueError("Holdings JSON did not contain any valid ticker entries.")
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
    ValueError
        If the input is missing required columns, contains invalid numbers, or
        requires prices that are unavailable.
    """

    if holdings_csv is not None:
        frame = _rename_known_columns(pd.read_csv(holdings_csv))
    elif holdings_mapping is not None:
        if holdings_kind is None:
            raise ValueError(
                "holdings_kind is required when holdings are provided as a mapping."
            )
        value_column = "current_value" if holdings_kind == "value" else "shares"
        frame = pd.DataFrame(
            {
                "ticker": list(holdings_mapping.keys()),
                value_column: list(holdings_mapping.values()),
            }
        )
    else:
        raise ValueError("Provide holdings via holdings_csv or holdings_mapping.")

    frame = _prepare_ticker_frame(frame)

    if holdings_kind is None:
        if "current_value" in frame.columns:
            resolved_kind: Literal["value", "shares"] = "value"
        elif "shares" in frame.columns:
            resolved_kind = "shares"
        else:
            raise ValueError(
                "Holdings CSV must include either a current_value/total_value column or a shares column."
            )
    else:
        resolved_kind = holdings_kind
        if resolved_kind == "value" and "current_value" not in frame.columns:
            data_columns = [column for column in frame.columns if column != "ticker"]
            if len(data_columns) == 1:
                frame = frame.rename(columns={data_columns[0]: "current_value"})
            else:
                raise ValueError("Holdings input is missing a current_value column.")
        if resolved_kind == "shares" and "shares" not in frame.columns:
            data_columns = [column for column in frame.columns if column != "ticker"]
            if len(data_columns) == 1:
                frame = frame.rename(columns={data_columns[0]: "shares"})
            else:
                raise ValueError("Holdings input is missing a shares column.")

    base = frame[["ticker"]].copy()
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
            raise ValueError(
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

    return base.groupby("ticker", as_index=False).agg(aggregation)


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
    FileNotFoundError
        If ``config_path`` is provided but does not exist.
    """

    if config_path is None:
        return AllocationConfig(), None

    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Allocation config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config_data = json.load(handle)

    fields = set(AllocationConfig.__dataclass_fields__)
    weights = config_data.get("allocation_weights", {})
    config = AllocationConfig(
        **{key: value for key, value in weights.items() if key in fields}
    )

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


def build_rebalance_plan(
    portfolio_name: str,
    *,
    new_cash: float,
    holdings_csv: Path | None = None,
    holdings_mapping: Mapping[str, float] | None = None,
    holdings_kind: Literal["value", "shares"] | None = None,
    export_dir: Path | None = None,
    config_path: Path | None = None,
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

    Returns
    -------
    RebalancePlan
        Fully prepared plan containing merged portfolio state, opportunity
        scores, and recommended buy orders.

    Raises
    ------
    ValueError
        If ``new_cash`` is not positive or the holdings input is invalid.
    FileNotFoundError
        If required optimiser artefacts or config files are missing.
    """

    if new_cash <= 0:
        raise ValueError("new_cash must be positive.")

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

    valid_prices = allocations["latest_price"].where(allocations["latest_price"] > 0)
    allocations["recommended_shares"] = (
        allocations["recommended_allocation"] / valid_prices
    )

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

    rows = plan.allocations.copy()
    if not include_zero_buys:
        rows = rows.loc[rows["recommended_allocation"] > 0].copy()

    if rows.empty:
        return "\n".join(
            [
                f"Rebalance plan for {plan.portfolio_name}",
                f"New cash: {_format_money(plan.new_cash)}",
                "",
                "No buys recommended.",
            ]
        )

    display = pd.DataFrame(
        {
            "Ticker": rows["ticker"],
            "Price": rows["latest_price"].map(_format_money),
            "Target": rows["target_weight"].map(_format_percent),
            "Current": rows["current_weight"].map(_format_percent),
            "Score": rows["opportunity_score"].map(lambda value: f"{value:.4f}"),
            "Buy $": rows["recommended_allocation"].map(_format_money),
            "Buy Shares": rows["recommended_shares"].map(_format_shares),
        }
    )

    return "\n".join(
        [
            f"Rebalance plan for {plan.portfolio_name}",
            f"New cash: {_format_money(plan.new_cash)}",
            "",
            display.to_string(index=False),
        ]
    )


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
