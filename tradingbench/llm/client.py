"""Minimal OpenAI-compatible chat client with tool calling.

No SDK on purpose: "OpenAI-compatible" providers diverge on the details, and
the harness has to see exactly what goes out and what comes back. The raw
response is kept on every Completion for the same reason.

Deliberately no retry, no streaming and no tokenisation: a retried call changes
the latency and token counts we are measuring, and cost is whatever the
provider reports, not an estimate computed here.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx


class LLMError(RuntimeError):
    """The provider refused the call or returned something unusable."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation requested by the model.

    `arguments` is the parsed form and `raw_arguments` the literal string the
    model produced; both are kept because scoring needs the parsed values while
    the replayable transcript needs the exact bytes.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""

    @property
    def parses(self) -> bool:
        """Whether the model produced understandable arguments.

        An empty argument object counts as unparsed: it is what a truncated or
        malformed JSON payload degrades to, and scoring must not read that as
        "the model called the tool with no arguments".
        """
        return bool(self.raw_arguments) and self.arguments != {}


@dataclass(slots=True)
class Completion:
    """One assistant turn, with the measurements attached to it."""

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
    """One provider: a base_url, a key, a model, and optional session headers."""

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
        # One id per client, not per request: the same session header has to
        # travel with every turn, and some gateways reject a call that has none
        # (see SESSION_HEADERS in providers).
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
        """Send one turn and return the parsed answer.

        Temperature defaults to 0 because the benchmark compares models, and
        sampling noise would be charged to the model. `max_tokens` caps one
        generation, so a model that rambles burns a bounded number of tokens per
        turn. HTTP and transport failures become LLMError: the caller counts them
        as a failed scenario rather than crashing the run.
        """
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

        # Every field is read defensively: compatible gateways omit `usage`,
        # `choices` or `model` often enough that a strict parse would report
        # their omission as a harness crash.
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
            # `cost` is a non-standard extension: only a few gateways report it,
            # and None means "unknown", never "free".
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
        arguments = {}  # unreadable arguments are a failed call, not an exception
    return ToolCall(
        id=str(call.get("id") or ""),
        name=str(function.get("name") or ""),
        arguments=arguments,
        raw_arguments=raw if isinstance(raw, str) else json.dumps(raw),
    )
