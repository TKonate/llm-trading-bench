"""Import des identifiants LLM depuis le magasin d'auth de Hermes.

Évite de recopier une clé à la main (et de la faire transiter par un chat ou un
historique shell). Ne fonctionne que si l'environnement dispose de ce fichier ;
sinon, renseigner LLM_BASE_URL / LLM_API_KEY / LLM_MODEL dans .env.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_AUTH_FILE = Path("/opt/data/profiles/jarvis/auth.json")

# Les modèles par défaut ne sont pas devinés : on reprend celui du profil.
DEFAULT_MODEL_BY_PROVIDER = {"opencode-go": "deepseek-v4.1-flash"}


def resolve_auth_file(explicit: Path | None, environ: dict[str, str] | None = None) -> Path:
    env = environ if environ is not None else os.environ
    if explicit:
        return Path(explicit)
    if env.get("HERMES_AUTH_JSON"):
        return Path(env["HERMES_AUTH_JSON"])
    if DEFAULT_AUTH_FILE.exists():
        return DEFAULT_AUTH_FILE
    return Path.home() / ".hermes" / "auth.json"


def read_credentials(provider: str, auth_file: Path) -> tuple[str, str]:
    """Retourne (base_url, access_token) sans jamais les écrire dans la sortie."""
    data = json.loads(auth_file.read_text(encoding="utf-8"))
    entries = (data.get("credential_pool") or {}).get(provider) or []
    usable = [entry for entry in entries if entry.get("access_token") and entry.get("base_url")]
    if not usable:
        raise SystemExit(
            f"Aucun identifiant exploitable pour « {provider} » dans {auth_file}. "
            "Renseigner LLM_BASE_URL / LLM_API_KEY à la main dans .env."
        )
    entry = sorted(usable, key=lambda item: item.get("priority", 0))[0]
    return str(entry["base_url"]), str(entry["access_token"])


def values_for_env(provider: str, auth_file: Path, model: str | None) -> dict[str, str]:
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
