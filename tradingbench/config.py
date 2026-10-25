"""Environment-driven configuration. No secrets in code, ever."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    broker: str = "mock"
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None
    alpaca_feed: str = "iex"
    log_dir: Path = PROJECT_ROOT / "logs"

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        log_dir = Path(env.get("TRADING_LOG_DIR", str(PROJECT_ROOT / "logs")))
        return cls(
            broker=env.get("TRADING_BROKER", "mock").strip().lower(),
            alpaca_api_key=(env.get("ALPACA_API_KEY") or "").strip() or None,
            alpaca_secret_key=(env.get("ALPACA_SECRET_KEY") or "").strip() or None,
            alpaca_feed=env.get("ALPACA_FEED", "iex").strip().lower(),
            log_dir=log_dir,
        )

    @property
    def has_alpaca_credentials(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)
