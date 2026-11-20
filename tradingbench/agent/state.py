"""An agent's persistent state: what has to survive a restart.

Three things must never be lost: the sessions already run (otherwise orders are
replayed), the consecutive-failure counter (otherwise the automatic pause never
fires) and the day's token spend (otherwise the budget cap is decorative).

Writes are atomic (temp file, then replace): a crash at the wrong moment must
not leave a truncated JSON behind, because a truncated state is indistinguishable
from a fresh start and would replay the slot.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AgentState:
    """Everything one agent remembers between sessions.

    `starting_equity_usd` is captured on the first session and never rewritten:
    it is the denominator every later performance figure is read against, so
    recomputing it would move the baseline of an ongoing experiment.
    """

    agent_id: str
    prompt_version: str = ""
    slots_done: list[str] = field(default_factory=list)
    total_trades: int = 0
    trades_since_critique: int = 0
    consecutive_tool_failures: int = 0
    tokens_spent_today: int = 0
    budget_day: str = ""
    plan: str = ""
    critique_count: int = 0
    recent_trades: list[dict[str, Any]] = field(default_factory=list)
    starting_equity_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view, with the two long lists truncated.

        The state file is read and rewritten every session, so it cannot grow
        with the experiment: 14 slots is a week of two sessions a day, and 20
        trades is four critique windows. Older entries are still in the JSONL
        log, which is where a post-mortem should look.
        """
        return {
            "agent_id": self.agent_id,
            "prompt_version": self.prompt_version,
            "slots_done": self.slots_done[-14:],
            "total_trades": self.total_trades,
            "trades_since_critique": self.trades_since_critique,
            "consecutive_tool_failures": self.consecutive_tool_failures,
            "tokens_spent_today": self.tokens_spent_today,
            "budget_day": self.budget_day,
            "plan": self.plan,
            "critique_count": self.critique_count,
            "recent_trades": self.recent_trades[-20:],
            "starting_equity_usd": self.starting_equity_usd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """Rebuild a state, ignoring keys this version does not know.

        Filtering on `__slots__` rather than on the dataclass fields keeps a
        state file written by a newer build readable by an older one instead of
        failing with an unexpected keyword argument.
        """
        return cls(**{key: data[key] for key in cls.__slots__ if key in data})  # type: ignore[arg-type]

    def is_slot_done(self, slot: str) -> bool:
        return slot in self.slots_done

    def mark_slot_done(self, slot: str) -> None:
        """Record a session as run. Idempotent: a slot key can only appear once."""
        if slot not in self.slots_done:
            self.slots_done.append(slot)

    def roll_day(self, today: date | None = None) -> None:
        """Reset the daily counters when the UTC date changes.

        The stored value is a plain ISO date rather than a timestamp, so the
        state file stays readable and the budget belongs to a day, not to an
        instant.
        """
        today = today or datetime.now(UTC).date()
        if self.budget_day != today.isoformat():
            self.budget_day = today.isoformat()
            self.tokens_spent_today = 0

    def record_trade(self, trade: dict[str, Any]) -> None:
        self.total_trades += 1
        self.trades_since_critique += 1
        self.recent_trades.append(trade)

    def record_call(self, tokens: int, tool_failures: int) -> None:
        """Book one session's cost and its tool failures.

        Failures accumulate across sessions and reset only on a call with no
        failure at all: three bad calls in a row pause the agent, an agent that
        fails once every ten sessions does not.
        """
        self.tokens_spent_today += tokens
        self.consecutive_tool_failures = (
            self.consecutive_tool_failures + tool_failures if tool_failures else 0
        )


def load_state(path: Path, agent_id: str) -> AgentState:
    """Read the state file, or start a fresh one.

    A corrupt file is renamed to `<name>.corrupt` rather than deleted: the next
    session runs on clean state, and the evidence of what went wrong is still on
    disk.
    """
    if not path.exists():
        return AgentState(agent_id=agent_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        path.rename(path.with_suffix(path.suffix + ".corrupt"))
        return AgentState(agent_id=agent_id)
    # The id in the file is only a hint: the caller decides which agent this is,
    # so a copied state file cannot rename an agent mid-experiment.
    data.setdefault("agent_id", agent_id)
    return AgentState.from_dict(data)


def save_state(path: Path, state: AgentState) -> None:
    """Write the state atomically, so a crash leaves either the old or the new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
