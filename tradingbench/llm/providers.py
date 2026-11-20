"""Resolve the LLM provider from the environment.

One shape, three uses: a hosted OpenAI-compatible provider, a local Ollama, or
any endpoint that speaks the same protocol. Whatever the case, the harness gets
a base_url, a model and optionally a key, so the client never learns which
vendor it is talking to.

Deliberately no model catalogue and no capability probing: a provider that
does not support tool calling fails the harness, which is the measurement we
want, not something to paper over with a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_BASE_URLS = {
    "ollama": "http://127.0.0.1:11434/v1",
    "opencode-go": "https://opencode.ai/zen/go/v1",
}

# Some gateways reject requests that carry no session identifier.
SESSION_HEADERS = {"opencode-go": "x-opencode-session"}


@dataclass(frozen=True)
class ProviderConfig:
    """Everything needed to build a client, and nothing that needs a network call."""

    name: str
    base_url: str
    model: str
    api_key: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    session_header: str | None = None

    @property
    def is_local(self) -> bool:
        """Whether traffic stays on this machine, which is what makes a run free."""
        return self.name == "ollama" or "127.0.0.1" in self.base_url or "localhost" in self.base_url

    def describe(self) -> str:
        """One-line summary for the CLI. Never includes the key."""
        where = "local" if self.is_local else self.base_url
        return f"{self.model} via {self.name} ({where})"


def from_env(environ: dict[str, str], model: str | None = None) -> ProviderConfig:
    """Build a provider config from environment variables.

    `model` overrides LLM_MODEL, which is how `tooleval --model a --model b`
    compares several models against one endpoint. Missing values raise
    ValueError instead of falling back to a default: an experiment silently run
    against the wrong model is worse than one that refuses to start.
    """
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
    # A key is mandatory only where a default base_url implies a paid gateway:
    # a local Ollama answers without one, and requiring it would break offline runs.
    if api_key is None and name == "opencode-go":
        raise ValueError("LLM_API_KEY manquant pour opencode-go (tradingbench importkeys)")
    return ProviderConfig(
        name=name,
        base_url=base_url,
        model=resolved_model,
        api_key=api_key,
        session_header=SESSION_HEADERS.get(name),
    )
