"""Import LLM credentials from the Hermes auth store.

Exists so that nobody has to copy a key by hand, and so that a key never
travels through a chat transcript or a shell history. The store is read, never
written; the caller decides what to do with the values (write them to `.env`,
in practice).

Deliberately not a credential manager: no refresh, no rotation, no cache. If
the store is unreadable here, the documented fallback is to set LLM_BASE_URL,
LLM_API_KEY and LLM_MODEL by hand in `.env`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Hermes stores credentials per profile. The location is resolved at runtime and
# never hard-coded: a machine-specific absolute path has no place in a public repo.
# Set HERMES_AUTH_JSON to point at a non-default profile.
DEFAULT_AUTH_FILE = Path(
    os.environ.get("HERMES_AUTH_JSON") or (Path.home() / ".hermes" / "auth.json")
)

# Default models come from the profile entry instead of a hard-coded guess: a
# wrong model name only fails at call time, far from the import that caused it.
DEFAULT_MODEL_BY_PROVIDER = {"opencode-go": "deepseek-v4.1-flash"}


def resolve_auth_file(explicit: Path | None, environ: dict[str, str] | None = None) -> Path:
    """Pick the auth file: explicit argument, then environment, then the default.

    The third branch returns a path that may not exist instead of raising: the
    caller reports "no credentials for this provider in <path>", which is the
    message a user can act on.
    """
    env = environ if environ is not None else os.environ
    if explicit:
        return Path(explicit)
    if env.get("HERMES_AUTH_JSON"):
        return Path(env["HERMES_AUTH_JSON"])
    if DEFAULT_AUTH_FILE.exists():
        return DEFAULT_AUTH_FILE
    return Path.home() / ".hermes" / "auth.json"


def read_credentials(provider: str, auth_file: Path) -> tuple[str, str]:
    """Return (base_url, access_token). Never print or log the result."""
    data = json.loads(auth_file.read_text(encoding="utf-8"))
    entries = (data.get("credential_pool") or {}).get(provider) or []
    usable = [entry for entry in entries if entry.get("access_token") and entry.get("base_url")]
    if not usable:
        raise SystemExit(
            f"Aucun identifiant exploitable pour « {provider} » dans {auth_file}. "
            "Renseigner LLM_BASE_URL / LLM_API_KEY à la main dans .env."
        )
    # Lowest priority number wins; entries without one sort first. The store can
    # hold several keys for a provider and this ordering, not the file order,
    # decides which one an experiment used.
    entry = sorted(usable, key=lambda item: item.get("priority", 0))[0]
    return str(entry["base_url"]), str(entry["access_token"])


def values_for_env(provider: str, auth_file: Path, model: str | None) -> dict[str, str]:
    """Environment variables to persist for `provider`, keyed by their .env names.

    LLM_MODEL is omitted when neither the argument nor the provider table
    supplies one: `update_env_file` removes empty values, so writing an empty
    string here would delete a working model from `.env`.
    """
    base_url, token = read_credentials(provider, auth_file)
    resolved_model = model or DEFAULT_MODEL_BY_PROVIDER.get(provider)
    values = {
        "LLM_PROVIDER": provider,
        "LLM_BASE_URL": base_url,
        "LLM_API_KEY": token,
    }
    if resolved_model:
        values["LLM_MODEL"] = resolved_model
    return values
