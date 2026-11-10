"""Tests for PySharpe configuration helpers.

.. note::

    **Canadian Investment Compliance** — Default MER values must be decimal
    fractions (< 0.10), never percentage points.  A real Canadian ETF MER
    expressed as a decimal fraction is always well below 0.10 (e.g., 0.0017
    for 0.17%).  Values like 0.17 signal the old percentage-point convention
    and are rejected.

    ``get_settings()`` is LRU-cached; call ``cache_clear()`` in tests
    that vary environment variables.  ``PYSHARPE_DATA_DIR`` overrides the
    data root.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from pysharpe import config, logging_utils


@pytest.fixture(autouse=True)
def clear_settings_cache():
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_build_settings_from_environment(monkeypatch, tmp_path):
    data_dir = tmp_path / "custom_data"
    log_dir = tmp_path / "custom_logs"
    monkeypatch.setenv("PYSHARPE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("PYSHARPE_LOG_DIR", str(log_dir))
    monkeypatch.setenv("PYSHARPE_LOG_LEVEL", "debug")
    monkeypatch.setenv("PYSHARPE_ARTIFACT_VERSION", "test")

    settings = config.build_settings()

    assert settings.data_dir == data_dir.resolve()
    assert settings.cache_dir == data_dir.resolve() / "cache"
    assert settings.log_dir == log_dir.expanduser()
    assert settings.log_level == "debug"
    assert settings.artifact_version == "test"


def test_get_settings_returns_cached_instance(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSHARPE_DATA_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    first = config.get_settings()
    second = config.get_settings()
    assert first is second
    assert isinstance(first.data_dir, Path)


def test_cwd_portfolio_config_does_not_change_library_defaults(monkeypatch, tmp_path):
    (tmp_path / "portfolio_config.json").write_text(
        json.dumps({"account_type": "TFSA", "account_room": {"TFSA": 1000}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert config.load_execution_config() == config.ExecutionConfig()
    assert config.get_settings().account_room == {}
    path = tmp_path / "portfolio_config.json"
    assert config.load_execution_config(path).account_type == "TFSA"
    assert config.build_settings(portfolio_config_path=path).account_room == {
        config.AccountType.TFSA: 1000
    }


def test_default_mer_by_ticker_values_are_decimal_fractions():
    """Default MER values must be decimal fractions, not percentage points.

    A real ETF MER expressed as a decimal fraction is always well below 0.10
    (i.e. below 10%). A value of 0.17 would indicate 17% MER — impossible for
    a listed ETF — which signals the old percentage-point convention.
    """
    settings = config.build_settings()
    for ticker, mer in settings.mer_by_ticker.items():
        assert mer < 0.10, (
            f"MER for {ticker} is {mer}, which looks like a percentage point "
            f"({mer * 100:.2f}%). Use decimal fractions (e.g. 0.0017 for 0.17%)."
        )


def test_ensure_directories_create_all(tmp_path):
    settings = config.build_settings(base_dir=tmp_path)
    # Ensure directories do not exist beforehand
    for path in (
        settings.cache_dir,
        settings.portfolio_dir,
        settings.price_history_dir,
        settings.export_dir,
        settings.info_dir,
        settings.log_dir,
    ):
        assert not path.exists()

    settings.ensure_directories()

    for path in (
        settings.cache_dir,
        settings.portfolio_dir,
        settings.price_history_dir,
        settings.export_dir,
        settings.info_dir,
        settings.log_dir,
    ):
        assert path.exists()


def test_configure_logging_uses_target_directory(tmp_path, monkeypatch):
    records: dict[str, object] = {}

    def fake_basicConfig(**kwargs):
        records.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basicConfig)

    root = logging.getLogger()
    existing_handlers = list(root.handlers)
    for handler in existing_handlers:
        root.removeHandler(handler)

    log_path = logging_utils.configure_logging(log_dir=tmp_path, level="debug")

    assert log_path.parent == tmp_path
    assert log_path.suffix == ".log"
    assert records.get("level") == "DEBUG"
    assert "filename" in records

    for handler in root.handlers:
        root.removeHandler(handler)
    for handler in existing_handlers:
        root.addHandler(handler)


# ---------------------------------------------------------------------------
# ExecutionConfig (tax / FX / fractional-share behaviour)
# ---------------------------------------------------------------------------


def test_execution_config_defaults():
    execution = config.ExecutionConfig()

    assert execution.account_type == "Non-Reg"
    assert execution.allow_fractional is True
    assert execution.fx_fee_bps == 0.0
    assert execution.fx_fee_decimal == 0.0
    assert execution.annual_tax_drag == 0.0


@pytest.mark.parametrize("account_type", ["TFSA", "FHSA", "tfsa", " fhsa "])
def test_execution_config_tax_drag_applies_to_tax_free_accounts(account_type):
    execution = config.ExecutionConfig(account_type=account_type)

    assert execution.tax_drag_applies is True
    assert execution.annual_tax_drag == pytest.approx(0.02 * 0.15)


@pytest.mark.parametrize("account_type", ["RRSP", "Non-Reg", "LIRA"])
def test_execution_config_tax_drag_skipped_for_other_accounts(account_type):
    execution = config.ExecutionConfig(account_type=account_type)

    assert execution.tax_drag_applies is False
    assert execution.annual_tax_drag == 0.0


def test_execution_config_fx_fee_decimal_is_basis_points():
    assert config.ExecutionConfig(fx_fee_bps=150).fx_fee_decimal == pytest.approx(0.015)


def test_load_execution_config_none_and_missing_path_return_defaults(tmp_path):
    assert config.load_execution_config() == config.ExecutionConfig()
    assert (
        config.load_execution_config(tmp_path / "absent.json")
        == config.ExecutionConfig()
    )


def test_load_execution_config_honours_known_keys_only(tmp_path):
    path = tmp_path / "portfolio_config.json"
    path.write_text(
        json.dumps(
            {
                "account_type": "TFSA",
                "allow_fractional": False,
                "fx_fee_bps": 150,
                "unknown_key": "ignored",
            }
        ),
        encoding="utf-8",
    )

    execution = config.load_execution_config(path)

    assert execution.account_type == "TFSA"
    assert execution.allow_fractional is False
    assert execution.fx_fee_bps == 150


def test_load_execution_config_invalid_json_warns_and_defaults(tmp_path, caplog):
    path = tmp_path / "portfolio_config.json"
    path.write_text("{not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        execution = config.load_execution_config(path)

    assert execution == config.ExecutionConfig()
    assert "Failed to load execution config" in caplog.text


def test_load_execution_config_unreadable_path_warns_and_defaults(tmp_path, caplog):
    path = tmp_path / "config_dir"
    path.mkdir()

    with caplog.at_level(logging.WARNING):
        execution = config.load_execution_config(path)

    assert execution == config.ExecutionConfig()
    assert "Failed to load execution config" in caplog.text


# ---------------------------------------------------------------------------
# AssetTaxProfile / withholding-tax matrix
# ---------------------------------------------------------------------------


def test_asset_tax_profile_rejects_negative_dividend_yield():
    with pytest.raises(ValueError, match="must be non-negative"):
        config.AssetTaxProfile(dividend_yield=-0.01)


def test_withholding_rate_for_cad_listed_us_equity_is_always_15_percent():
    profile = config.AssetTaxProfile(is_cad_listed_us_equity=True, dividend_yield=0.012)

    for account in config.AccountType:
        assert config.calculate_withholding_tax_rate(account, profile) == 0.15


def test_withholding_rate_for_non_us_asset_is_zero():
    profile = config.AssetTaxProfile(dividend_yield=0.03)

    for account in config.AccountType:
        assert config.calculate_withholding_tax_rate(account, profile) == 0.0


@pytest.mark.parametrize(
    ("account", "expected"),
    [
        (config.AccountType.RRSP, 0.0),
        (config.AccountType.TFSA, 0.15),
        (config.AccountType.FHSA, 0.15),
        (config.AccountType.NON_REG, 0.0),
        (config.AccountType.LIRA, 0.0),
        (config.AccountType.RRIF, 0.0),
    ],
)
def test_withholding_rate_for_us_listed_asset_by_account(account, expected):
    profile = config.AssetTaxProfile(is_us_listed=True, dividend_yield=0.013)

    assert config.calculate_withholding_tax_rate(account, profile) == expected


# ---------------------------------------------------------------------------
# Ticker metadata heuristics
# ---------------------------------------------------------------------------


def test_ticker_metadata_for_cad_listed_ticker():
    assert config.get_ticker_metadata("vfv.to") == {
        "is_us_domiciled": False,
        "is_cad_denominated": True,
    }


def test_ticker_metadata_for_usd_suffix_ticker():
    assert config.get_ticker_metadata("VOO-USD") == {
        "is_us_domiciled": True,
        "is_cad_denominated": False,
    }


def test_ticker_metadata_fallback_marks_ticker_non_cad():
    assert config.get_ticker_metadata("VOO") == {
        "is_us_domiciled": False,
        "is_cad_denominated": False,
    }


def test_ticker_metadata_proxy_map_takes_precedence():
    proxy_map = {"VFV.TO": {"is_us_domiciled": True, "is_cad_denominated": False}}

    assert config.get_ticker_metadata("VFV.TO", proxy_map=proxy_map) == {
        "is_us_domiciled": True,
        "is_cad_denominated": False,
    }


def test_ticker_metadata_proxy_entry_defaults_to_cad():
    proxy_map = {"VFV.TO": {}}

    assert config.get_ticker_metadata("VFV.TO", proxy_map=proxy_map) == {
        "is_us_domiciled": False,
        "is_cad_denominated": True,
    }


def test_ticker_metadata_ignores_proxy_map_miss():
    proxy_map = {"VCN.TO": {"is_cad_denominated": True}}

    assert config.get_ticker_metadata("VFV.TO", proxy_map=proxy_map) == {
        "is_us_domiciled": False,
        "is_cad_denominated": True,
    }


# ---------------------------------------------------------------------------
# build_settings: MER map, proxy map, and portfolio config
# ---------------------------------------------------------------------------


def test_build_settings_reads_mer_map_from_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYSHARPE_MER_BY_TICKER", json.dumps({"VFV.TO": 0.0016}))

    settings = config.build_settings()

    assert settings.mer_by_ticker == {"VFV.TO": 0.0016}
    assert all(mer < 0.10 for mer in settings.mer_by_ticker.values())


def test_build_settings_invalid_mer_json_falls_back(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYSHARPE_MER_BY_TICKER", "{not json")

    with caplog.at_level(logging.WARNING):
        settings = config.build_settings()

    assert settings.mer_by_ticker == {"VEQT": 0.0017}
    assert "not a valid JSON string" in caplog.text


def test_build_settings_non_numeric_mer_falls_back(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYSHARPE_MER_BY_TICKER", json.dumps({"VFV.TO": "cheap"}))

    with caplog.at_level(logging.WARNING):
        settings = config.build_settings()

    assert settings.mer_by_ticker == {"VEQT": 0.0017}
    assert "Invalid values in PYSHARPE_MER_BY_TICKER" in caplog.text


def test_build_settings_loads_proxy_map_from_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "proxy_map.json").write_text(
        json.dumps({"VFV.TO": {"is_cad_denominated": True}}), encoding="utf-8"
    )

    settings = config.build_settings()

    assert settings.proxy_map == {"VFV.TO": {"is_cad_denominated": True}}


def test_build_settings_invalid_proxy_map_warns(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "proxy_map.json").write_text("{not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        settings = config.build_settings()

    assert settings.proxy_map == {}
    assert "Failed to load proxy_map.json" in caplog.text


def test_build_settings_reads_tax_profile_from_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "PYSHARPE_TAX_PROFILE",
        json.dumps({"marginal_tax_rate": 0.45, "capital_gains_inclusion_rate": 0.5}),
    )

    settings = config.build_settings()

    assert settings.tax_profile.marginal_tax_rate == 0.45


def test_build_settings_invalid_tax_profile_json_warns(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYSHARPE_TAX_PROFILE", "[1, 2]")

    with caplog.at_level(logging.WARNING):
        settings = config.build_settings()

    assert settings.tax_profile.marginal_tax_rate == 0.40
    assert "PYSHARPE_TAX_PROFILE is not valid JSON" in caplog.text


def test_build_settings_reads_account_capacities_from_environment(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYSHARPE_ACCOUNT_CAPACITIES", json.dumps({"TFSA": 95000}))

    settings = config.build_settings()

    assert settings.account_capacities == {config.AccountType.TFSA: 95000.0}


def test_build_settings_skips_invalid_account_capacity_entries(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "PYSHARPE_ACCOUNT_CAPACITIES",
        json.dumps({"TFSA": 95000, "RRSP": "lots"}),
    )

    with caplog.at_level(logging.WARNING):
        settings = config.build_settings()

    assert settings.account_capacities == {config.AccountType.TFSA: 95000.0}
    assert "Skipping invalid account_capacity entry" in caplog.text


def test_build_settings_invalid_capacity_json_warns(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYSHARPE_ACCOUNT_CAPACITIES", "{not json")

    with caplog.at_level(logging.WARNING):
        settings = config.build_settings()

    assert settings.account_capacities == {}
    assert "PYSHARPE_ACCOUNT_CAPACITIES is not valid JSON" in caplog.text


def test_build_settings_parses_portfolio_config_sections(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    portfolio_config = tmp_path / "portfolio_config.json"
    portfolio_config.write_text(
        json.dumps(
            {
                "account_room": {"TFSA": 7000, "RRSP": 30000},
                "tax_profile": {
                    "marginal_tax_rate": 0.43,
                    "capital_gains_inclusion_rate": 0.5,
                },
                "account_capacities": {"TFSA": 95000},
                "asset_tax_profiles": {
                    "VOO": {"is_us_listed": True, "dividend_yield": 0.013},
                    "VFV.TO": {
                        "is_cad_listed_us_equity": True,
                        "dividend_yield": 0.012,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    settings = config.build_settings(portfolio_config_path=portfolio_config)

    assert settings.account_room == {
        config.AccountType.TFSA: 7000.0,
        config.AccountType.RRSP: 30000.0,
    }
    assert settings.account_capacities == {config.AccountType.TFSA: 95000.0}
    assert settings.tax_profile.marginal_tax_rate == 0.43
    assert settings.asset_tax_profiles["VOO"].is_us_listed is True
    assert settings.asset_tax_profiles["VFV.TO"].is_cad_listed_us_equity is True


def test_build_settings_skips_invalid_portfolio_config_entries(tmp_path, caplog):
    portfolio_config = tmp_path / "portfolio_config.json"
    portfolio_config.write_text(
        json.dumps(
            {
                "account_room": {"TFSA": 7000, "RRSP": "lots"},
                "tax_profile": {"marginal_tax_rate": "high"},
                "account_capacities": {"FHSA": "lots"},
                "asset_tax_profiles": {
                    "VOO": {"is_us_listed": True, "dividend_yield": -0.01}
                },
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        settings = config.build_settings(portfolio_config_path=portfolio_config)

    assert settings.account_room == {config.AccountType.TFSA: 7000.0}
    assert settings.account_capacities == {}
    assert settings.tax_profile.marginal_tax_rate == 0.40
    assert settings.asset_tax_profiles == {}
    assert "Skipping invalid account_room entry" in caplog.text
    assert "Skipping invalid tax_profile field" in caplog.text
    assert "Skipping invalid account_capacities entry" in caplog.text
    assert "Skipping invalid asset_tax_profile" in caplog.text


def test_build_settings_invalid_portfolio_config_warns(tmp_path, caplog):
    portfolio_config = tmp_path / "portfolio_config.json"
    portfolio_config.write_text("{not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        settings = config.build_settings(portfolio_config_path=portfolio_config)

    assert settings.account_room == {}
    assert "Failed to load account config" in caplog.text


def test_settings_resolves_relative_log_dir_under_data_dir(tmp_path):
    settings = config.PySharpeSettings(data_dir=tmp_path, log_dir=Path("custom-logs"))

    assert settings.log_dir == (tmp_path.resolve() / "custom-logs")
