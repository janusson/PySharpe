"""Central configuration utilities for PySharpe."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_DEFAULT_DATA_DIR = Path("data")
_DEFAULT_PORTFOLIO_CONFIG_PATH = Path("portfolio_config.json")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionConfig:
    """Per-portfolio execution settings for tax, FX, and share handling.

    Attributes
    ----------
    account_type : str
        Account registration type. ``"TFSA"`` and ``"FHSA"`` trigger a 15%
        withholding-tax drag on US-domiciled securities during expected-return
        estimation. ``"RRSP"`` and ``"Non-Reg"`` apply no tax drag.
        Default is ``"Non-Reg"``.
    allow_fractional : bool
        When ``True``, recommended share counts may be fractional. When
        ``False``, shares are floored to whole units (with leftover cash
        tracked separately). Default is ``True``.
    fx_fee_bps : float
        Foreign-exchange conversion fee in basis points applied to non-
        CAD-denominated purchases during execution. For example, 150 = 1.5%.
        Default is ``0.0``.
    dividend_yield_estimate : float
        Estimated annual dividend yield used to compute the withholding-tax
        drag for US-domiciled assets. Default is ``0.02`` (2%).
    withholding_tax_rate : float
        US withholding-tax rate applied to dividend yield for accounts that
        do not have treaty protection (TFSA, FHSA). Default is ``0.15``.
    norberts_commission : float
        Flat commission per trade when executing Norbert's Gambit (e.g.,
        buying DLR.TO and selling DLR-U.TO). Two commissions are charged
        per round-trip. Default is ``6.95`` (typical Canadian discount
        brokerage).
    norberts_spread_bps : float
        Bid-ask spread of the dual-listed security (e.g., DLR.TO / DLR-U.TO)
        expressed in basis points. Default is ``2.0`` (0.02%).
    norberts_drift_risk_bps : float
        Estimated adverse market-drift risk during the 2--3 business day
        journaling period, expressed in basis points of the transaction
        value. Default is ``5.0`` (0.05%).
    norberts_dlr_price_cad : float
        Approximate share price of DLR.TO in CAD, used to compute the
        number of shares in the execution checklist. Default is ``14.0``.
    """

    account_type: str = "Non-Reg"
    allow_fractional: bool = True
    fx_fee_bps: float = 0.0
    dividend_yield_estimate: float = 0.02
    withholding_tax_rate: float = 0.15
    norberts_commission: float = 6.95
    norberts_spread_bps: float = 2.0
    norberts_drift_risk_bps: float = 5.0
    norberts_dlr_price_cad: float = 14.0

    @property
    def tax_drag_applies(self) -> bool:
        """Return True when the account type triggers US withholding-tax drag."""
        return self.account_type.upper() in {"TFSA", "FHSA"}

    @property
    def fx_fee_decimal(self) -> float:
        """Return the FX fee as a decimal fraction (e.g. 150 bps → 0.015)."""
        return self.fx_fee_bps / 10000.0

    @property
    def annual_tax_drag(self) -> float:
        """Annual expected-return reduction from US withholding tax.

        Computed as ``dividend_yield_estimate * withholding_tax_rate`` when
        ``tax_drag_applies`` is True; otherwise returns 0.0.
        """
        if not self.tax_drag_applies:
            return 0.0
        return self.dividend_yield_estimate * self.withholding_tax_rate

    @property
    def norberts_spread_decimal(self) -> float:
        """Return the Norbert's Gambit spread as a decimal fraction."""
        return self.norberts_spread_bps / 10000.0

    @property
    def norberts_drift_decimal(self) -> float:
        """Return the Norbert's Gambit drift risk as a decimal fraction."""
        return self.norberts_drift_risk_bps / 10000.0


def load_execution_config(
    config_path: Path | None = None,
) -> ExecutionConfig:
    """Load execution settings from a portfolio_config.json, with sensible defaults.

    Parameters
    ----------
    config_path : Path or None
        Path to a ``portfolio_config.json`` file. When ``None``, the file
        ``portfolio_config.json`` in the current working directory is used if
        it exists; otherwise defaults are returned.

    Returns
    -------
    ExecutionConfig
    """
    path = Path(config_path) if config_path else _DEFAULT_PORTFOLIO_CONFIG_PATH
    if not path.exists():
        return ExecutionConfig()

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load execution config from %s: %s", path, exc)
        return ExecutionConfig()

    kwargs: dict = {}
    for key in (
        "account_type",
        "allow_fractional",
        "fx_fee_bps",
        "norberts_commission",
        "norberts_spread_bps",
        "norberts_drift_risk_bps",
        "norberts_dlr_price_cad",
    ):
        if key in data:
            kwargs[key] = data[key]
    return ExecutionConfig(**kwargs)


def get_ticker_metadata(
    ticker: str,
    proxy_map: dict[str, dict[str, object]] | None = None,
) -> dict[str, bool]:
    """Return ``is_us_domiciled`` and ``is_cad_denominated`` for a ticker.

    Looks up the ticker in ``proxy_map`` first; falls back to sensible
    defaults based on ticker suffix heuristics:

    - ``.TO`` suffix → Canadian-domiciled, CAD-denominated.
    - ``-USD`` suffix or no ``.TO`` → US-domiciled, non-CAD.

    Parameters
    ----------
    ticker : str
        The ticker symbol to query.
    proxy_map : dict or None
        The proxy map loaded by :func:`build_settings`. When ``None`` an
        empty map is assumed.

    Returns
    -------
    dict[str, bool]
        Dictionary with keys ``"is_us_domiciled"`` and
        ``"is_cad_denominated"``.
    """
    if proxy_map and ticker in proxy_map:
        entry = proxy_map[ticker]
        return {
            "is_us_domiciled": bool(entry.get("is_us_domiciled", False)),
            "is_cad_denominated": bool(entry.get("is_cad_denominated", True)),
        }

    # Heuristic fallback
    if ticker.upper().endswith(".TO"):
        return {"is_us_domiciled": False, "is_cad_denominated": True}

    if "-USD" in ticker.upper():
        return {"is_us_domiciled": True, "is_cad_denominated": False}

    return {"is_us_domiciled": False, "is_cad_denominated": False}


@dataclass(frozen=True)
class PySharpeSettings:
    """Application-level settings with directory layout and behaviour toggles.

    Attributes:
        data_dir: Root directory for all generated artefacts.
        cache_dir: Folder containing cached remote data such as DuckDB price history.
        price_history_dir: Folder containing per-ticker CSV downloads.
        export_dir: Folder containing collated portfolios and optimisation outputs.
        portfolio_dir: Folder containing user-provided portfolio definitions.
        info_dir: Folder used by the info download helpers.
        log_dir: Folder for timestamped log files.
        log_level: Default logging threshold (INFO by default).
        artifact_version: Metadata version tag written to exports.

    Example:
        >>> from pysharpe.config import build_settings
        >>> settings = build_settings()
        >>> settings.export_dir.name
        'exports'
    """

    data_dir: Path = field(default_factory=lambda: _DEFAULT_DATA_DIR)
    cache_dir: Path = field(init=False)
    price_history_dir: Path = field(init=False)
    export_dir: Path = field(init=False)
    portfolio_dir: Path = field(init=False)
    info_dir: Path = field(init=False)
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    log_level: str = "INFO"
    artifact_version: str = "v1"
    mer_by_ticker: dict[str, float] = field(default_factory=dict)
    proxy_map: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "data_dir", self.data_dir.resolve())
        object.__setattr__(self, "cache_dir", self.data_dir / "cache")
        object.__setattr__(self, "portfolio_dir", self.data_dir / "portfolio")
        object.__setattr__(self, "price_history_dir", self.data_dir / "price_hist")
        object.__setattr__(self, "export_dir", self.data_dir / "exports")
        object.__setattr__(self, "info_dir", self.data_dir / "info")
        log_dir = self.log_dir
        if not log_dir.is_absolute():
            log_dir = (self.data_dir / log_dir).resolve()
        else:
            log_dir = log_dir.resolve()
        object.__setattr__(self, "log_dir", log_dir)

    def ensure_directories(self) -> None:
        """Create the standard directory tree if it is missing.

        Example:
            >>> from pysharpe.config import build_settings
            >>> settings = build_settings()
            >>> settings.ensure_directories()  # Creates folders on disk
        """

        for directory in (
            self.cache_dir,
            self.portfolio_dir,
            self.price_history_dir,
            self.export_dir,
            self.info_dir,
            self.log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _path_from_env(var_name: str, default: Path) -> Path:
    override = os.getenv(var_name)
    if not override:
        return default
    return Path(override).expanduser()


def build_settings(base_dir: Path | None = None) -> PySharpeSettings:
    """Construct settings, honouring environment overrides when present.

    Args:
        base_dir: Optional override for the root data directory. When omitted the
            function respects the ``PYSHARPE_DATA_DIR`` environment variable or
            defaults to ``./data``.

    Returns:
        A fully initialised :class:`PySharpeSettings` instance.

    Example:
        >>> from pathlib import Path
        >>> from pysharpe.config import build_settings
        >>> custom = build_settings(base_dir=Path('tmp-data'))
        >>> custom.data_dir.name
        'tmp-data'
    """

    if base_dir is None:
        data_dir = _path_from_env("PYSHARPE_DATA_DIR", _DEFAULT_DATA_DIR)
        data_dir = data_dir.expanduser().resolve()
        log_dir_env = os.getenv("PYSHARPE_LOG_DIR")
        if log_dir_env:
            log_dir = Path(log_dir_env).expanduser().resolve()
        else:
            log_dir = data_dir / "logs"
    else:
        data_dir = Path(base_dir).expanduser().resolve()
        log_dir = data_dir / "logs"

    log_level = os.getenv("PYSHARPE_LOG_LEVEL", "INFO")
    artifact_version = os.getenv("PYSHARPE_ARTIFACT_VERSION", "v1")

    mer_by_ticker_str = os.getenv("PYSHARPE_MER_BY_TICKER")
    if mer_by_ticker_str:
        try:
            mer_by_ticker = json.loads(mer_by_ticker_str)
            # Ensure keys are strings and values are floats
            if not all(
                isinstance(k, str) and isinstance(v, (int, float))
                for k, v in mer_by_ticker.items()
            ):
                raise ValueError(
                    "MER_BY_TICKER values must be string to float mapping."
                )
            mer_by_ticker = {k: float(v) for k, v in mer_by_ticker.items()}
        except json.JSONDecodeError:
            logger.warning(
                "PYSHARPE_MER_BY_TICKER is not a valid JSON string. Using default MERs."
            )
            mer_by_ticker = {"VEQT": 0.0017}
        except ValueError as e:
            logger.warning(
                f"Invalid values in PYSHARPE_MER_BY_TICKER: {e}. Using default MERs."
            )
            mer_by_ticker = {"VEQT": 0.0017}
    else:
        mer_by_ticker = {"VEQT": 0.0017}

    proxy_map = {}
    proxy_map_path = Path("proxy_map.json")
    if proxy_map_path.exists():
        try:
            with open(proxy_map_path) as f:
                proxy_map = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load proxy_map.json: {e}")

    return PySharpeSettings(
        data_dir=data_dir,
        log_dir=log_dir,
        log_level=log_level,
        artifact_version=artifact_version,
        mer_by_ticker=mer_by_ticker,
        proxy_map=proxy_map,
    )


@lru_cache(maxsize=1)
def get_settings() -> PySharpeSettings:
    """Return the cached settings instance for the current process.

    Returns:
        The singleton :class:`PySharpeSettings` created via :func:`build_settings`.

    Example:
        >>> from pysharpe.config import get_settings
        >>> first = get_settings()
        >>> second = get_settings()
        >>> first is second
        True
    """

    return build_settings()


__all__ = [
    "ExecutionConfig",
    "PySharpeSettings",
    "build_settings",
    "get_settings",
    "get_ticker_metadata",
    "load_execution_config",
]
