"""Tests for .env handling: comments must survive, secrets must not leak."""

from __future__ import annotations

import stat
from pathlib import Path

from dotenv import dotenv_values

from tradingbench.setkeys import mask, update_env_file


def test_update_preserves_comments_and_other_keys(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("# mon commentaire\nTRADING_BROKER=mock\nALPACA_API_KEY=\n")
    update_env_file(env, {"ALPACA_API_KEY": "PKTEST123", "ALPACA_SECRET_KEY": "secret456"})
    content = env.read_text()
    values = dotenv_values(env)
    assert "# mon commentaire" in content
    assert values["TRADING_BROKER"] == "mock"
    assert values["ALPACA_API_KEY"] == "PKTEST123"
    assert values["ALPACA_SECRET_KEY"] == "secret456"


def test_file_is_private(tmp_path: Path):
    env = tmp_path / ".env"
    update_env_file(env, {"ALPACA_API_KEY": "k"})
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_empty_value_removes_the_line(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("ALPACA_API_KEY=old\n")
    update_env_file(env, {"ALPACA_API_KEY": ""})
    assert "ALPACA_API_KEY" not in env.read_text()


def test_mask_never_reveals_the_middle():
    assert mask("PKABCDEFGHIJKLMNOPQR") == "PKAB…OPQR (20 car.)"
    assert mask("short") == "*****"
    assert mask("") == "(vide)"
