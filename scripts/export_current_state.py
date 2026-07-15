"""Export current-state CSVs for ``pysharpe allocate``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Ensure pysharpe is importable from the scripts/ directory.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pysharpe.exceptions import DataIngestionError, DataValidationError

logger = logging.getLogger("export_current_state")


def _load_holdings_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "ticker" not in frame.columns:
        raise DataValidationError("Holdings CSV must include a ticker column.")
    frame = frame[["ticker"] + [col for col in frame.columns if col != "ticker"]]
    return frame


def _load_holdings_json(raw: str) -> pd.Series:
    candidate = Path(raw).expanduser()
    payload = candidate.read_text(encoding="utf-8") if candidate.exists() else raw
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise DataValidationError("Holdings JSON must be a ticker -> number mapping.")
    return pd.Series(data, name="shares").rename_axis("ticker")


def _fetch_latest_price(ticker: str) -> float:
    history = yf.download(
        ticker,
        period="5d",
        interval="1d",
        progress=False,
        threads=False,
    )
    if history.empty:
        raise DataIngestionError(f"No price history for {ticker}.")
    return float(history["Close"].iloc[-1])


def _resolve_latest_prices(tickers: pd.Series) -> pd.Series:
    prices = {}
    for ticker in sorted(tickers.dropna().unique()):
        prices[ticker] = _fetch_latest_price(ticker)
    return pd.Series(prices, name="latest_price")


def _load_weights(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if "ticker" not in frame.columns:
        raise DataValidationError("Weights file must include a ticker column.")
    if "target_weight" not in frame.columns:
        if "weight" in frame.columns:
            frame = frame.rename(columns={"weight": "target_weight"})
        else:
            raise DataValidationError(
                "Weights file must include target_weight or weight."
            )
    return (
        frame[["ticker", "target_weight"]]
        .assign(
            target_weight=lambda x: pd.to_numeric(x["target_weight"], errors="coerce")
        )
        .dropna(subset=["target_weight"])
        .set_index("ticker")["target_weight"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="export_current_state.py",
        description="Export ticker/current_value/target_weight CSV for pysharpe allocate.",
    )
    parser.add_argument(
        "--holdings-csv", type=Path, help="Holdings CSV with ticker and shares."
    )
    parser.add_argument("--holdings-json", help="JSON mapping ticker -> shares.")
    parser.add_argument(
        "--weights", required=True, type=Path, help="Optimiser weights file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("current_state.csv"),
        help="Output path for the exported current state CSV.",
    )

    args = parser.parse_args()
    if (args.holdings_csv is None) == (args.holdings_json is None):
        parser.error("Provide either --holdings-csv or --holdings-json.")

    try:
        if args.holdings_csv:
            holdings = _load_holdings_csv(args.holdings_csv).set_index("ticker")
            if "shares" not in holdings.columns:
                raise DataValidationError(
                    "Holdings CSV must include a shares column."
                )
            shares = holdings["shares"].astype(float)
        else:
            shares = _load_holdings_json(args.holdings_json).astype(float)

        prices = _resolve_latest_prices(shares.index.to_series())
        values = shares * prices.reindex(shares.index)

        weights = _load_weights(args.weights)

        frame = pd.DataFrame(
            {
                "ticker": shares.index,
                "shares": shares.values,
                "latest_price": prices.reindex(shares.index).values,
                "current_value": values.values,
                "target_weight": weights.reindex(shares.index).fillna(0.0).values,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame[["ticker", "current_value", "target_weight"]].to_csv(
            args.output, index=False
        )
        logger.info("Wrote %s", args.output)
    except (DataIngestionError, DataValidationError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    raise SystemExit(main())
