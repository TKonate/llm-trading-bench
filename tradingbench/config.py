"""Environment-driven configuration: the single place settings are read.

Every value comes from the process environment (optionally a `.env` file that
the CLI has already loaded), so the benchmark can be reproduced by recording
the environment rather than by editing code. The module deliberately does no
validation beyond trimming and casting: a malformed number raises here, at
startup, instead of being quietly replaced by a default nobody notices.

The Alpaca credentials enter the process here, and nothing else in the package
reads ALPACA_* from the environment. No value is ever logged or printed here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The project root anchors every relative path in the environment: the CLI is
# launched by absolute path from cron, where the working directory is arbitrary.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Resolved settings. Defaults are the mock broker, i.e. no network, no key."""

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
        """Starting capital converted once, and the basis of every performance figure.

        Alpaca offers no EUR account, so the experiment runs in USD against a
        reference capital converted at a single rate. FX is then ignored for the
        rest of the run: that is a documented simplification, not an oversight,
        and it must not be re-applied per session or the percentages drift.
        """
        return round(self.starting_capital_eur * self.eur_usd, 2)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        """Build settings from `environ`, defaulting to the real environment.

        The injection point exists for tests and for replaying a recorded
        environment; passing `{}` yields the defaults, which is how the mock
        path stays runnable with no shell setup at all.
        """
        env = os.environ if environ is None else environ
        raw_log_dir = Path(env.get("TRADING_LOG_DIR", "logs")).expanduser()
        # A relative path must resolve against the project, not against the
        # caller's working directory.
        log_dir = (
            raw_log_dir if raw_log_dir.is_absolute() else (PROJECT_ROOT / raw_log_dir).resolve()
        )
        raw_veille = (env.get("VEILLE_PATH") or "").strip()
        veille_path = Path(raw_veille).expanduser() if raw_veille else None
        return cls(
            broker=env.get("TRADING_BROKER", "mock").strip().lower(),
            # Empty strings are treated as absent: `KEY=` in a .env is how a key
            # gets disabled, and it must not look like a credential to the
            # broker factory.
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
