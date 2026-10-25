"""Résolution du provider LLM depuis l'environnement.

Un seul format, trois usages : un provider hébergé compatible OpenAI, Ollama en
local, ou n'importe quel endpoint qui parle le même protocole.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_BASE_URLS = {
    "ollama": "http://127.0.0.1:11434/v1",
    "opencode-go": "https://opencode.ai/zen/go/v1",
}

# Certains gateways refusent les requêtes sans identifiant de session.
SESSION_HEADERS = {"opencode-go": "x-opencode-session"}


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    session_header: str | None = None

    @property
    def is_local(self) -> bool:
        return self.name == "ollama" or "127.0.0.1" in self.base_url or "localhost" in self.base_url

    def describe(self) -> str:
        where = "local" if self.is_local else self.base_url
        return f"{self.model} via {self.name} ({where})"


def from_env(environ: dict[str, str], model: str | None = None) -> ProviderConfig:
    name = (environ.get("LLM_PROVIDER") or "ollama").strip().lower()
    base_url = (environ.get("LLM_BASE_URL") or DEFAULT_BASE_URLS.get(name, "")).strip()
    if not base_url:
        raise ValueError(
            f"LLM_BASE_URL manquant pour le provider {name!r} "
            f"(connus : {', '.join(sorted(DEFAULT_BASE_URLS))})"
        )
    resolved_model = (model or environ.get("LLM_MODEL") or "").strip()
    if not resolved_model:
        raise ValueError("LLM_MODEL manquant : indiquer le modèle à tester")
    api_key = (environ.get("LLM_API_KEY") or "").strip() or None
    if api_key is None and name == "opencode-go":
        raise ValueError("LLM_API_KEY manquant pour opencode-go (tradingbench importkeys)")
    return ProviderConfig(
        name=name,
        base_url=base_url,
        model=resolved_model,
        api_key=api_key,
        session_header=SESSION_HEADERS.get(name),
    )
