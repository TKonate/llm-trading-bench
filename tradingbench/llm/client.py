"""Client chat minimal, compatible OpenAI, avec tool calling.

Volontairement sans SDK : les providers « compatibles OpenAI » divergent sur les
détails, et le harnais doit voir exactement ce qui part et ce qui revient.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx


class LLMError(RuntimeError):
    """Le provider a refusé ou cassé l'appel."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""

    @property
    def parses(self) -> bool:
        return bool(self.raw_arguments) and self.arguments != {}


@dataclass(slots=True)
class Completion:
    text: str
    tool_calls: list[ToolCall]
    model: str
    finish_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatClient:
    """Un provider = une base_url + une clé + un modèle (+ en-têtes éventuels)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        session_header: str | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        user_agent: str = "tradingbench/0.1",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._session_id = str(uuid.uuid4())
        headers = {"User-Agent": user_agent, "Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if session_header:
            headers[session_header] = self._session_id
        headers.update(extra_headers or {})
        self._client = httpx.Client(timeout=timeout, headers=headers, transport=transport)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        started = time.perf_counter()
        try:
            response = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.model}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code >= 400:
            raise LLMError(f"{self.model}: HTTP {response.status_code} {response.text[:300]}")

        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}

        return Completion(
            text=message.get("content") or "",
            tool_calls=[_parse_tool_call(call) for call in message.get("tool_calls") or []],
            model=data.get("model") or self.model,
            finish_reason=choice.get("finish_reason") or "unknown",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost_usd=data.get("cost") if isinstance(data.get("cost"), int | float) else None,
            latency_ms=latency_ms,
            raw=data,
        )

    def close(self) -> None:
        self._client.close()


def _parse_tool_call(call: dict) -> ToolCall:
    function = call.get("function") or {}
    raw = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except json.JSONDecodeError:
        arguments = {}  # arguments illisibles : c'est un échec, pas une exception
    return ToolCall(
        id=str(call.get("id") or ""),
        name=str(function.get("name") or ""),
        arguments=arguments,
        raw_arguments=raw if isinstance(raw, str) else json.dumps(raw),
    )
