"""Run the harness and aggregate its results.

The runner owns the multi-turn loop and the per-model summary. It deliberately
computes no score of its own: pass/fail comes from `scenarios.score`, so the
admissibility rule lives in one place and a reader can audit it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from ..jsonlog import JsonlLogger
from ..llm.client import Completion, ToolCall
from .executor import ToolExecutor
from .scenarios import SCENARIOS, SYSTEM_PROMPT, TOOLS, Scenario, score


class Completer(Protocol):
    """The slices of ChatClient the runner needs.

    Declared here rather than imported as ChatClient so a test double (or a
    recorded transcript) can be scored without an HTTP layer.
    """

    model: str

    def complete(self, messages, *, tools=None, temperature=0.0, max_tokens=800) -> Completion: ...


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Outcome of one scenario for one model, ready to be logged or tabulated.

    `error` set means the call itself failed, which is distinct from a wrong
    answer: both are `ok=False`, but only the first is the harness's problem.
    """

    model: str
    scenario_id: str
    ok: bool
    reason: str
    called: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    total_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "scenario": self.scenario_id,
            "ok": self.ok,
            "reason": self.reason,
            "called": self.called,
            "latency_ms": round(self.latency_ms, 1),
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "error": self.error,
        }


def run_scenarios(
    client: Completer,
    scenarios: tuple[Scenario, ...] = SCENARIOS,
    *,
    logger: JsonlLogger | None = None,
    max_turns: int = 4,
    executor_factory: Callable[[], ToolExecutor] = ToolExecutor,
) -> list[ScenarioResult]:
    """Run every scenario, with the tools actually called, over several turns.

    An agent reads the price before deciding, so it is given the time to do so;
    otherwise the harness measures first-token reactivity, not task ability.
    Each scenario gets its own broker, so the positions of one cannot contaminate
    the next.

    `latency` and `tokens` accumulate over all turns of a scenario, not per turn:
    the cost of an answer includes the detour through the tools, which is
    precisely what a model that needs three calls pays for.
    """
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": scenario.prompt},
        ]
        executor = executor_factory()
        if scenario.seed:
            executor.seed(scenario.seed)
        all_calls: list[ToolCall] = []
        latency = 0.0
        tokens = 0

        try:
            for _ in range(max_turns):
                completion = client.complete(messages, tools=TOOLS)
                latency += completion.latency_ms
                tokens += completion.total_tokens
                if not completion.tool_calls:
                    break
                all_calls.extend(completion.tool_calls)
                messages.append(_assistant_message(completion))
                for call in completion.tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": executor.dispatch(call.name, call.arguments),
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - a broken model is a result
            result = ScenarioResult(
                model=getattr(client, "model", "?"),
                scenario_id=scenario.id,
                ok=False,
                reason="appel LLM en échec",
                called=[call.name for call in all_calls],
                latency_ms=latency,
                total_tokens=tokens,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            outcome = score(scenario, all_calls)
            result = ScenarioResult(
                model=getattr(client, "model", "?"),
                scenario_id=scenario.id,
                ok=outcome.ok,
                reason=outcome.reason,
                called=[call.name for call in all_calls],
                latency_ms=latency,
                total_tokens=tokens,
            )
        results.append(result)
        if logger:
            logger.log("tooleval", **result.as_dict())
    return results


def _assistant_message(completion: Completion) -> dict:
    """Replay the assistant turn in the transcript, with the raw argument strings."""
    return {
        "role": "assistant",
        "content": completion.text or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.raw_arguments},
            }
            for call in completion.tool_calls
        ],
    }


@dataclass(frozen=True, slots=True)
class Summary:
    """Per-model aggregate. `total` counts scenarios, not turns."""

    model: str
    passed: int
    total: int
    median_latency_ms: float
    total_tokens: int

    @property
    def pass_rate(self) -> float:
        return 0.0 if self.total == 0 else self.passed / self.total

    @property
    def verdict(self) -> str:
        """Eliminatory: below 100 %, the model does not join the benchmark."""
        return "GO" if self.passed == self.total else "NO-GO"


def summarize(results: list[ScenarioResult]) -> list[Summary]:
    """Aggregate per model, ordered by pass rate then by speed.

    The "median" is the middle element of the sorted latencies, i.e. the upper
    median for an even count, not the average of the two middle values: the
    figure is compared across models, not used in a statistical test. Scenarios
    that errored are excluded from the latency sample, because a timeout is not a
    latency.
    """
    by_model: dict[str, list[ScenarioResult]] = {}
    for result in results:
        by_model.setdefault(result.model, []).append(result)

    summaries: list[Summary] = []
    for model, items in by_model.items():
        latencies = sorted(item.latency_ms for item in items if not item.error)
        median = latencies[len(latencies) // 2] if latencies else 0.0
        summaries.append(
            Summary(
                model=model,
                passed=sum(1 for item in items if item.ok),
                total=len(items),
                median_latency_ms=median,
                total_tokens=sum(item.total_tokens for item in items),
            )
        )
    return sorted(summaries, key=lambda summary: (-summary.pass_rate, summary.median_latency_ms))
