"""Central configuration utilities for PySharpe."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path

_DEFAULT_DATA_DIR = Path("data")
_DEFAULT_PORTFOLIO_CONFIG_PATH = Path("portfolio_config.json")

logger = logging.getLogger(__name__)


class AccountType(Enum):
    """Canadian registered account types with distinct tax treatments.

    Attributes:
        TFSA: Tax-Free Savings Account — no tax on gains, but no treaty
            protection for US withholding tax on US-domiciled assets.
        RRSP: Registered Retirement Savings Plan — exempt from US
            withholding tax on directly-held US-domiciled assets under the
            Canada-U.S. tax treaty.
        FHSA: First Home Savings Account — similar to TFSA for withholding
            tax purposes (no treaty protection).
        NON_REGISTERED: Taxable (cash/margin) account — US withholding tax
            is generally recoverable via the Foreign Tax Credit.
    """

    TFSA = "TFSA"
    RRSP = "RRSP"
    FHSA = "FHSA"
    NON_REGISTERED = "NON_REGISTERED"


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


@dataclass(frozen=True)
class AssetTaxProfile:
    """Tax-relevant characteristics of a single investable asset.

    Used in conjunction with :func:`calculate_withholding_tax_rate` to
    determine the unrecoverable US withholding-tax drag for an asset held
    in a given Canadian account type.

    Attributes:
        is_us_listed: ``True`` for US-domiciled ETFs such as VOO or ITOT.
        is_cad_listed_us_equity: ``True`` for Canadian-listed wrappers of
            US equities such as VFV.TO (which holds VOO) or QQC.TO.
        dividend_yield: The trailing 12-month dividend yield expressed as a
            decimal (e.g. ``0.013`` for 1.3 %).

    Example:
        >>> voo = AssetTaxProfile(is_us_listed=True, dividend_yield=0.013)
        >>> vfv = AssetTaxProfile(is_cad_listed_us_equity=True, dividend_yield=0.012)
    """

    is_us_listed: bool = False
    is_cad_listed_us_equity: bool = False
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.dividend_yield < 0.0:
            raise ValueError(
                f"dividend_yield must be non-negative, got {self.dividend_yield}"
            )


def calculate_withholding_tax_rate(
    account_type: AccountType,
    tax_profile: AssetTaxProfile,
) -> float:
    """Compute the unrecoverable US withholding-tax rate for an asset in an account.

    Implements the following Canadian tax logic:

    * **RRSP + US-listed**: ``0.0`` — exempt under the Canada-U.S. tax treaty
      for directly-held US-domiciled assets.
    * **TFSA / FHSA + US-listed**: ``0.15`` — 15 % withholding tax applies;
      these accounts have no treaty protection.
    * **CAD-listed US equity**: ``0.15`` regardless of account type — the
      foreign withholding tax is lost at the fund level and is unrecoverable.
    * **Non-registered**: ``0.0`` — assumed fully offset by the Canadian
      Foreign Tax Credit for the purposes of drag calculation.
    * **Non-US assets**: ``0.0`` — no US withholding tax applies.

    Parameters:
        account_type: The Canadian registered account type.
        tax_profile: The asset's tax-relevant characteristics.

    Returns:
        The withholding tax rate as a decimal (e.g. ``0.15`` for 15 %).

    Example:
        >>> voo = AssetTaxProfile(is_us_listed=True, dividend_yield=0.013)
        >>> calculate_withholding_tax_rate(AccountType.RRSP, voo)
        0.0
        >>> calculate_withholding_tax_rate(AccountType.TFSA, voo)
        0.15
    """
    # CAD-listed US equity: withholding is lost at the fund level regardless
    if tax_profile.is_cad_listed_us_equity:
        return 0.15

    # Non-US assets: no US withholding tax applies
    if not tax_profile.is_us_listed:
        return 0.0

    # US-listed assets — account-type-specific treatment
    if account_type == AccountType.RRSP:
        return 0.0  # Treaty-protected
    if account_type in (AccountType.TFSA, AccountType.FHSA):
        return 0.15  # No treaty protection
    if account_type == AccountType.NON_REGISTERED:
        return 0.0  # FTC offsets

    return 0.0


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
    account_room: dict[AccountType, float] = field(default_factory=dict)
    asset_tax_profiles: dict[str, AssetTaxProfile] = field(default_factory=dict)

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

    # Load account_room and asset_tax_profiles from portfolio_config.json
    account_room: dict[AccountType, float] = {}
    asset_tax_profiles: dict[str, AssetTaxProfile] = {}
    portfolio_config_path = Path("portfolio_config.json")
    if portfolio_config_path.exists():
        try:
            with portfolio_config_path.open("r", encoding="utf-8") as f:
                portfolio_data = json.load(f)

            # account_room: {"TFSA": 7000, "RRSP": 30000, ...}
            raw_room = portfolio_data.get("account_room")
            if raw_room and isinstance(raw_room, dict):
                for key, value in raw_room.items():
                    try:
                        acct = AccountType(key.upper())
                        account_room[acct] = float(value)
                    except (ValueError, TypeError) as exc:
                        logger.warning(
                            "Skipping invalid account_room entry '%s': %s", key, exc
                        )

            # asset_tax_profiles: {"VOO": {"is_us_listed": true, ...}, ...}
            raw_profiles = portfolio_data.get("asset_tax_profiles")
            if raw_profiles and isinstance(raw_profiles, dict):
                for ticker, profile_data in raw_profiles.items():
                    try:
                        asset_tax_profiles[ticker] = AssetTaxProfile(
                            is_us_listed=bool(profile_data.get("is_us_listed", False)),
                            is_cad_listed_us_equity=bool(
                                profile_data.get("is_cad_listed_us_equity", False)
                            ),
                            dividend_yield=float(
                                profile_data.get("dividend_yield", 0.0)
                            ),
                        )
                    except (ValueError, TypeError) as exc:
                        logger.warning(
                            "Skipping invalid asset_tax_profile for '%s': %s",
                            ticker,
                            exc,
                        )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to load account config from %s: %s",
                portfolio_config_path,
                exc,
            )

    return PySharpeSettings(
        data_dir=data_dir,
        log_dir=log_dir,
        log_level=log_level,
        artifact_version=artifact_version,
        mer_by_ticker=mer_by_ticker,
        proxy_map=proxy_map,
        account_room=account_room,
        asset_tax_profiles=asset_tax_profiles,
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
    "AccountType",
    "AssetTaxProfile",
    "ExecutionConfig",
    "PySharpeSettings",
    "build_settings",
    "calculate_withholding_tax_rate",
    "get_settings",
    "get_ticker_metadata",
    "load_execution_config",
]
