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
    starting_capital_eur: float = 1000.0
    eur_usd: float = 1.08
    veille_path: Path | None = None

    @property
    def budget_usd(self) -> float:
        """Capital de départ converti, base de toutes les mesures de performance.

        Alpaca ne propose pas de compte en euros : on travaille en USD avec un
        capital de référence converti une fois pour toutes. Le taux est fixé à
        l'ouverture de l'expérience (FX ignoré ensuite, et c'est documenté).
        """
        return round(self.starting_capital_eur * self.eur_usd, 2)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        raw_log_dir = Path(env.get("TRADING_LOG_DIR", "logs")).expanduser()
        # A relative path must resolve against the project, not against the
        # caller's working directory: the CLI is launched by absolute path.
        log_dir = (
            raw_log_dir if raw_log_dir.is_absolute() else (PROJECT_ROOT / raw_log_dir).resolve()
        )
        raw_veille = (env.get("VEILLE_PATH") or "").strip()
        veille_path = Path(raw_veille).expanduser() if raw_veille else None
        return cls(
            broker=env.get("TRADING_BROKER", "mock").strip().lower(),
            alpaca_api_key=(env.get("ALPACA_API_KEY") or "").strip() or None,
            alpaca_secret_key=(env.get("ALPACA_SECRET_KEY") or "").strip() or None,
            alpaca_feed=env.get("ALPACA_FEED", "iex").strip().lower(),
            log_dir=log_dir,
            starting_capital_eur=float(env.get("STARTING_CAPITAL_EUR", "1000")),
            eur_usd=float(env.get("EURUSD_RATE", "1.08")),
            veille_path=veille_path,
        )

    @property
    def has_alpaca_credentials(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)
