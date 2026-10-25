"""Settings must not depend on the caller's working directory."""

from __future__ import annotations

from pathlib import Path

from tradingbench.config import PROJECT_ROOT, Settings


def test_relative_log_dir_resolves_against_project():
    settings = Settings.from_env({"TRADING_LOG_DIR": "./logs"})
    assert settings.log_dir.is_absolute()
    assert settings.log_dir == (PROJECT_ROOT / "logs").resolve()


def test_absolute_log_dir_is_kept(tmp_path: Path):
    settings = Settings.from_env({"TRADING_LOG_DIR": str(tmp_path)})
    assert settings.log_dir == tmp_path


def test_blank_credentials_become_none():
    settings = Settings.from_env({"ALPACA_API_KEY": "   ", "ALPACA_SECRET_KEY": ""})
    assert settings.alpaca_api_key is None
    assert settings.has_alpaca_credentials is False


def test_default_broker_is_the_offline_mock():
    assert Settings.from_env({}).broker == "mock"
